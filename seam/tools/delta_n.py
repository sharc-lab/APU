"""ΔN - does enabling a compute backend reduce the maximum context the device can hold?

Three arms. In every one of them the measured generation runs on cpu-p; what varies is whether
the iGPU also has the model compiled and resident while that generation happens. A is cpu-p
alone, B adds the resident iGPU, and A_prime repeats A as an independent arm so the comparison
carries its own noise floor. A_prime runs last, so its agreement with A brackets B in time and
any drift over the session is charged to the noise floor rather than to the accelerator.

Each repeat runs in its own child process (``seam.tools._delta_n_child``). At and beyond the
ceiling the failure arrives as an allocation throw, an access violation, or the OS killing the
process, and a child confines all three to one repeat. Process exit is also the most complete
teardown available between rungs.

Two distinctions the ladder depends on:

* A **ceiling failure** is a child that loaded its backends and then failed to generate. That is
  evidence about context capacity.
* An **inadmissible block** is a quiescence refusal, canary drift, a failure before the model
  was resident, or a repeat whose wall-clock window contains Kernel-Power 506/507 (Modern
  Standby enter/exit). That is evidence about the machine, not the ceiling, and it is retried.
  Letting one masquerade as the other would put the ceiling wherever the background load
  happened to be. A standby-spanning repeat is not a slow measurement; it is not a measurement.

The paging gate is reporting-only here: a rung that pages is still a rung that completed or did
not, so page reads are recorded on every block and never used to discard a completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seam.backends.local_openvino import runtime_info
from seam.config import ResolvedConfig, resolve_config
from seam.errors import SeamError
from seam.manifest import emit
from seam.measurement import MeasurementRefusalError, machine_measurement
from seam.model_provenance import load_local_spec, manifest_model_block
from seam.powerstate import capture_power_state, manifest_power_state
from seam.rawstore import RunDir
from seam.tools.prompt_a_lifecycle import assert_acyclic, open_in_progress_run

_PLATFORM_PATH = Path("configs/platforms/aipc-c1.yaml")
_MEASUREMENT_PATH = Path("configs/measurement.yaml")
_DELTA_N_PATH = Path("configs/delta_n.yaml")

MB = 1024.0 * 1024.0


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "ts": _utc(), **fields}, sort_keys=True), flush=True)


# ------------------------------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------------------------------


def build_exact_prompt(tokenizer: Any, *, target_tokens: int, unit: str) -> tuple[str, int]:
    """Render a chat prompt that tokenizes to exactly ``target_tokens``.

    The rendered string is what every arm receives byte-for-byte, so prompt length can never
    become a function of which backend is enabled.
    """

    def render(content: str) -> str:
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        )

    def count(content: str) -> int:
        return len(tokenizer(render(content))["input_ids"])

    if count("") > target_tokens:
        raise SeamError(
            f"chat-template scaffold alone exceeds the requested {target_tokens} tokens"
        )

    unit_tokens = max(1, len(tokenizer(unit)["input_ids"]))
    low, high = 0, (target_tokens // unit_tokens) + 2
    while low < high:
        mid = (low + high + 1) // 2
        if count(unit * mid) <= target_tokens:
            low = mid
        else:
            high = mid - 1
    content = unit * low

    # Close the remainder a token at a time. Batching the estimate first keeps this to a couple
    # of passes even at 40k.
    pad = " a"
    for _ in range(64):
        current = count(content)
        if current == target_tokens:
            return render(content), current
        if current > target_tokens:
            raise SeamError(f"overshot while padding to {target_tokens}: reached {current}")
        content += pad * (target_tokens - current)

    final = count(content)
    raise SeamError(
        f"could not construct an exact {target_tokens}-token prompt; stalled at {final}"
    )


def prompt_for(
    *, root: Path, tokenizer: Any, n_tokens: int, unit: str, cache: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Build (once) and cache the exact-N prompt shared by every arm."""
    if n_tokens in cache:
        return cache[n_tokens]
    out_dir = root / "derived" / "delta_n" / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"n{n_tokens}.txt"
    text, realized = build_exact_prompt(tokenizer, target_tokens=n_tokens, unit=unit)
    path.write_text(text, encoding="utf-8")
    entry = {
        "n_tokens_requested": n_tokens,
        "n_tokens_realized": realized,
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }
    cache[n_tokens] = entry
    _log("delta_n.prompt_built", **entry)
    return entry


# ------------------------------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------------------------------


def observe_processes(names: list[str]) -> dict[str, Any]:
    """Record the platform's own AI stack. A control, never varied."""
    import psutil

    found: dict[str, Any] = {}
    wanted = {name.lower() for name in names}
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            raw = str(process.info.get("name") or "")
            stem = raw.lower().removesuffix(".exe")
            if stem in wanted:
                found.setdefault(stem, []).append(
                    {
                        "pid": int(process.info["pid"]),
                        "rss_mb": float(process.info["memory_info"].rss) / MB,
                    }
                )
        except Exception:  # a process that exits mid-scan is not an error
            continue
    return {
        "observed_utc": _utc(),
        "requested": sorted(wanted),
        "present": {name: found.get(name, []) for name in sorted(wanted)},
        "total_rss_mb": sum(row["rss_mb"] for rows in found.values() for row in rows),
        "role": "control_not_factor",
    }


def memory_now() -> dict[str, float]:
    import psutil

    virtual = psutil.virtual_memory()
    return {
        "available_mb": virtual.available / MB,
        "used_mb": virtual.used / MB,
        "percent": virtual.percent,
    }


def wait_for_recovery(cfg: dict[str, Any]) -> dict[str, Any]:
    """Let the OS reclaim what a dead child was holding before the next block reads memory."""
    recovery = cfg["recovery"]
    settle_s = float(recovery["settle_s"])
    floor_mb = float(recovery["min_available_mb"])
    deadline = time.monotonic() + float(recovery["max_wait_s"])
    time.sleep(settle_s)
    waited = settle_s
    while memory_now()["available_mb"] < floor_mb and time.monotonic() < deadline:
        time.sleep(float(recovery["poll_s"]))
        waited += float(recovery["poll_s"])
    final = memory_now()
    return {
        "settle_s": settle_s,
        "waited_s": waited,
        "floor_mb": floor_mb,
        "available_mb_after": final["available_mb"],
        "reached_floor": final["available_mb"] >= floor_mb,
    }


# ------------------------------------------------------------------------------------------
# One repeat
# ------------------------------------------------------------------------------------------


def run_child(
    *,
    root: Path,
    work_dir: Path,
    spec: dict[str, Any],
    timeout_s: float,
    tag: str,
) -> dict[str, Any]:
    """Run one repeat in an isolated process and classify how it ended."""
    spec_path = work_dir / f"{tag}.spec.json"
    out_path = work_dir / f"{tag}.result.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    if out_path.exists():
        out_path.unlink()

    argv = [
        sys.executable,
        "-u",
        "-m",
        "seam.tools._delta_n_child",
        "--spec",
        str(spec_path),
        "--out",
        str(out_path),
    ]
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        returncode = completed.returncode
        stderr_tail = (completed.stderr or "")[-3000:]
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stderr_tail = str(exc.stderr or "")[-3000:]
    wall_s = time.perf_counter() - started

    payload: dict[str, Any] = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"phase": "unreadable_result_file"}

    phase = str(payload.get("phase", "no_result_file"))
    reached_resident = bool(payload.get("memory_resident_before_inference"))
    completed_ok = bool(payload.get("completed"))

    if completed_ok:
        outcome, failure_mode = "pass", None
    elif timed_out:
        outcome, failure_mode = "fail", f"timeout:{timeout_s:.0f}s@{phase}"
    elif payload.get("failure_mode"):
        outcome = "fail" if reached_resident else "load_failure"
        failure_mode = str(payload["failure_mode"])
    elif returncode not in (0, None):
        # Windows reports an access violation as 0xC0000005 / -1073741819. A child that dies
        # without raising is the OS refusing an allocation, which is a ceiling observation.
        outcome = "fail" if reached_resident else "load_failure"
        failure_mode = f"process_died:returncode={returncode}@{phase}"
    else:
        outcome, failure_mode = "load_failure", f"no_result@{phase}"

    record = {
        "tag": tag,
        "outcome": outcome,
        "failure_mode": failure_mode,
        "phase_reached": phase,
        "reached_resident": reached_resident,
        "returncode": returncode,
        "timed_out": timed_out,
        "parent_wall_s": wall_s,
        "stderr_tail": stderr_tail if outcome != "pass" else "",
        "child": payload,
    }
    _log(
        "delta_n.repeat",
        tag=tag,
        outcome=outcome,
        failure_mode=failure_mode,
        phase=phase,
        wall_s=round(wall_s, 2),
    )
    return record


def _reject_kernel_power_ids(cfg: dict[str, Any]) -> set[int]:
    adm = cfg.get("admissibility") or {}
    if adm.get("modern_standby_inadmissible") is False:
        return set()
    raw = adm.get("reject_kernel_power_ids")
    if raw is None:
        return set()
    return {int(x) for x in raw}


def _modern_standby_hits(
    *,
    cfg: dict[str, Any],
    started_utc: str,
    ended_utc: str,
) -> list[dict[str, Any]]:
    """Kernel-Power events from the pre-registered reject set inside the attempt window."""
    reject_ids = _reject_kernel_power_ids(cfg)
    if not reject_ids:
        return []
    from seam.tools.acceptance_instrumentation import collect_kernel_power_events

    query = collect_kernel_power_events(start_utc=started_utc, end_utc=ended_utc)
    hits: list[dict[str, Any]] = []
    for event in query.get("events") or []:
        eid = event.get("id")
        if eid is None:
            continue
        if int(eid) in reject_ids:
            hits.append(event)
    return hits


def measured_repeat(
    *,
    root: Path,
    cfg: dict[str, Any],
    p_cpus: list[int],
    work_dir: Path,
    child_spec: dict[str, Any],
    label: str,
    tag: str,
) -> dict[str, Any]:
    """Wrap one repeat in the machine-lock / quiescence / canary envelope, retrying refusals."""
    max_retries = int(cfg["admissibility"]["max_block_retries"])
    attempts: list[dict[str, Any]] = []

    for attempt in range(max_retries):
        recovery = wait_for_recovery(cfg)
        attempt_started_utc = _utc()
        try:
            with machine_measurement(
                repo_root=root,
                label=label,
                config=cfg,
                p_cpus=p_cpus,
                sample_paging=True,
            ) as block:
                result = run_child(
                    root=root,
                    work_dir=work_dir,
                    spec=child_spec,
                    timeout_s=float(cfg["generation"]["timeout_s"]),
                    tag=f"{tag}.a{attempt}",
                )
            envelope = block.record
        except MeasurementRefusalError as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "admissible": False,
                    "reason": "quiescence_refusal",
                    "detail": str(exc)[:600],
                    "started_utc": attempt_started_utc,
                    "ended_utc": _utc(),
                    "recovery": recovery,
                }
            )
            _log("delta_n.block_refused", tag=tag, attempt=attempt, reason=str(exc)[:300])
            continue

        attempt_ended_utc = _utc()
        env = _compact_envelope(envelope)
        # Prefer the lock envelope for the coincidence window when present.
        window_start = env.get("lock_acquired_utc") or attempt_started_utc
        window_end = env.get("lock_released_utc") or attempt_ended_utc

        # Mechanistic: any reject-list Kernel-Power id in the window makes the attempt
        # inadmissible regardless of how fast or slow the child looked. Recorded verbatim.
        standby_hits = _modern_standby_hits(cfg=cfg, started_utc=window_start, ended_utc=window_end)
        if standby_hits:
            attempts.append(
                {
                    "attempt": attempt,
                    "admissible": False,
                    "reason": "modern_standby_in_window",
                    "detail": (
                        f"Kernel-Power ids {[int(e['id']) for e in standby_hits]} in "
                        f"[{window_start}, {window_end}]"
                    ),
                    "started_utc": attempt_started_utc,
                    "ended_utc": attempt_ended_utc,
                    "window_utc": {
                        "start": window_start,
                        "end": window_end,
                        "source": (
                            "envelope.lock_acquired/released_utc"
                            if env.get("lock_acquired_utc") and env.get("lock_released_utc")
                            else "harness_attempt_wall_clock"
                        ),
                    },
                    "kernel_power_events": standby_hits,
                    "result": result,
                    "envelope": env,
                    "recovery": recovery,
                }
            )
            _log(
                "delta_n.modern_standby_inadmissible",
                tag=tag,
                attempt=attempt,
                event_ids=[int(e["id"]) for e in standby_hits],
                window_start=window_start,
                window_end=window_end,
            )
            continue

        # A load failure says nothing about the ceiling; retry it rather than record a ceiling.
        if result["outcome"] == "load_failure":
            child_exc = (result.get("child") or {}).get("exception") or {}
            exception_message = child_exc.get("message")
            attempts.append(
                {
                    "attempt": attempt,
                    "admissible": False,
                    "reason": "load_failure",
                    "detail": result["failure_mode"],
                    "exception_message": exception_message,
                    "started_utc": attempt_started_utc,
                    "ended_utc": attempt_ended_utc,
                    "recovery": recovery,
                }
            )
            _log(
                "delta_n.load_failure",
                tag=tag,
                attempt=attempt,
                failure_mode=result["failure_mode"],
                exception_message=exception_message,
            )
            continue

        return {
            "admissible": True,
            "attempts": attempts,
            "attempt_used": attempt,
            "recovery": recovery,
            "result": result,
            "envelope": env,
            "started_utc": attempt_started_utc,
            "ended_utc": attempt_ended_utc,
            "window_utc": {
                "start": window_start,
                "end": window_end,
                "source": (
                    "envelope.lock_acquired/released_utc"
                    if env.get("lock_acquired_utc") and env.get("lock_released_utc")
                    else "harness_attempt_wall_clock"
                ),
            },
        }

    raise SeamError(_ceiling_refusal_message(label, attempts, max_retries))


def _ceiling_refusal_message(label: str, attempts: list[dict[str, Any]], max_retries: int) -> str:
    """Distinguish deterministic load/exception failures from quiescence/contention."""
    reasons = [a["reason"] for a in attempts]
    n_standby = sum(1 for r in reasons if r == "modern_standby_in_window")
    if attempts and n_standby * 2 >= len(attempts):
        return (
            f"{label}: {max_retries} consecutive inadmissible blocks ({reasons}); "
            f"all or most were modern_standby_in_window (Kernel-Power 506/507 in the "
            f"attempt window). PowerRequestSystemRequired did not keep the machine awake, "
            f"or the assertion was not held. Stop and report - this is a platform fact."
        )
    n_deterministic = sum(1 for r in reasons if r == "load_failure")
    # "most" includes a tie: 2/3 or 2/2 or 3/3 - not 1/3.
    if attempts and n_deterministic * 2 >= len(attempts):
        details = [
            a.get("exception_message") or a.get("detail")
            for a in attempts
            if a["reason"] == "load_failure"
        ]
        return (
            f"{label}: {max_retries} consecutive inadmissible blocks ({reasons}); "
            f"all or most were deterministic load_failure/exception "
            f"(details={details}); not a quiescence/contention finding. Stop and report."
        )
    return (
        f"{label}: {max_retries} consecutive inadmissible blocks "
        f"({reasons}); refusing to record a ceiling from a machine "
        "that will not hold still. Stop and report."
    )


def _compact_envelope(record: dict[str, Any]) -> dict[str, Any]:
    """Keep the validity evidence; drop the per-sample arrays that would bloat the seal."""
    quiescence = record.get("quiescence") or {}
    pressure = record.get("memory_pressure") or {}
    return {
        "label": record.get("label"),
        "valid": record.get("valid"),
        "invalid_reasons": record.get("invalid_reasons"),
        "lock_wait_s": record.get("lock_wait_s"),
        "lock_acquired_utc": record.get("lock_acquired_utc"),
        "lock_released_utc": record.get("lock_released_utc"),
        "quiescence_passed": quiescence.get("passed"),
        "quiescence_aggregate": quiescence.get("aggregate"),
        "canary_relative_drift": record.get("canary_relative_drift"),
        "canary_gate_verdict": (record.get("canary_gate") or {}).get("verdict"),
        "paging_gate_verdict": (record.get("paging_gate") or {}).get("verdict"),
        "paging_gate_reasons": (record.get("paging_gate") or {}).get("reasons"),
        "available_memory_mb_before": record.get("available_memory_mb_before"),
        "available_memory_mb_after": record.get("available_memory_mb_after"),
        "available_memory_mb_min": pressure.get("available_memory_mb_min"),
        "hard_page_reads_max_per_s": (
            max(
                (r for r in (record.get("hard_page_reads_per_s") or []) if r is not None),
                default=None,
            )
        ),
    }


# ------------------------------------------------------------------------------------------
# Ladder
# ------------------------------------------------------------------------------------------


def run_rung(
    *,
    root: Path,
    cfg: dict[str, Any],
    p_cpus: list[int],
    work_dir: Path,
    run_dir: RunDir,
    arm: dict[str, Any],
    model_dir: str,
    prompt: dict[str, Any],
    n_tokens: int,
    stage: str,
) -> dict[str, Any]:
    """Run every repeat at one rung. PASS requires all of them."""
    ladder = cfg["ladder"]
    repeats = int(ladder["repeats"])
    affinity = list(p_cpus) if cfg["openvino"]["affinity"] == "p_cpus" else None
    cpu_properties = dict(cfg["openvino"]["cpu_properties"])

    load_sequence = [
        {
            "device": device,
            "properties": cpu_properties if device == "CPU" else {},
        }
        for device in arm["load_sequence"]
    ]
    child_spec = {
        "model_dir": model_dir,
        "load_sequence": load_sequence,
        "generate_device": arm["generate_device"],
        "prompt_path": prompt["path"],
        "max_new_tokens": int(ladder["max_new_tokens"]),
        "affinity_cpus": affinity,
        "rss_interval_s": 0.05,
    }

    records: list[dict[str, Any]] = []
    for repeat in range(repeats):
        tag = f"{arm['id']}.n{n_tokens}.r{repeat}"
        record = measured_repeat(
            root=root,
            cfg=cfg,
            p_cpus=p_cpus,
            work_dir=work_dir,
            child_spec=child_spec,
            label=f"delta-n/{arm['id']}/{n_tokens}/{repeat}",
            tag=tag,
        )
        record["arm_id"] = arm["id"]
        record["n_tokens"] = n_tokens
        record["repeat_index"] = repeat
        record["stage"] = stage
        record["prompt"] = prompt
        record["controls"] = observe_processes(cfg["controls"]["observe_processes"])
        records.append(record)
        run_dir.append_ndjson("repeats.ndjson", record)

    n_pass = sum(1 for r in records if r["result"]["outcome"] == "pass")
    if n_pass == repeats:
        verdict = "PASS"
    elif n_pass == 0:
        verdict = "FAIL"
    else:
        verdict = "MIXED"

    rung = {
        "arm_id": arm["id"],
        "n_tokens": n_tokens,
        "stage": stage,
        "repeats": repeats,
        "n_pass": n_pass,
        "verdict": verdict,
        "failure_modes": sorted(
            {r["result"]["failure_mode"] for r in records if r["result"]["failure_mode"]}
        ),
        "generation_wall_s": [
            (r["result"]["child"].get("generation") or {}).get("wall_s") for r in records
        ],
        "peak_rss_mb": [
            ((r["result"]["child"].get("generation") or {}).get("peak_rss_bytes") or 0) / MB or None
            for r in records
        ],
        "free_memory_mb_min": [
            (r["result"]["child"].get("generation") or {}).get("free_memory_mb_min")
            for r in records
        ],
        "standing_reservation": [r["result"]["child"].get("standing_reservation") for r in records],
        "paging_gate_verdicts": [r["envelope"]["paging_gate_verdict"] for r in records],
        "canary_drift": [r["envelope"]["canary_relative_drift"] for r in records],
    }
    _log(
        "delta_n.rung",
        arm=arm["id"],
        n=n_tokens,
        verdict=verdict,
        n_pass=n_pass,
        stage=stage,
        failure_modes=rung["failure_modes"],
    )
    run_dir.append_ndjson("rungs.ndjson", rung)
    return rung


def run_arm(
    *,
    root: Path,
    cfg: dict[str, Any],
    p_cpus: list[int],
    work_dir: Path,
    run_dir: RunDir,
    arm: dict[str, Any],
    model_dir: str,
    tokenizer: Any,
    prompt_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Ascending ladder to first non-PASS, then bisect to the configured resolution."""
    ladder = cfg["ladder"]
    unit = str(ladder["filler_unit"])
    resolution = int(ladder["bisect_resolution_tokens"])
    round_to = int(ladder["bisect_round_to"])

    started = _utc()
    controls_at_start = observe_processes(cfg["controls"]["observe_processes"])
    memory_at_start = memory_now()
    _log(
        "delta_n.arm_start",
        arm=arm["id"],
        label=arm["label"],
        available_mb=round(memory_at_start["available_mb"], 1),
        control_rss_mb=round(controls_at_start["total_rss_mb"], 1),
    )

    rungs: list[dict[str, Any]] = []
    highest_pass = 0
    lowest_nonpass: int | None = None

    for n_tokens in [int(v) for v in ladder["rungs"]]:
        prompt = prompt_for(
            root=root, tokenizer=tokenizer, n_tokens=n_tokens, unit=unit, cache=prompt_cache
        )
        rung = run_rung(
            root=root,
            cfg=cfg,
            p_cpus=p_cpus,
            work_dir=work_dir,
            run_dir=run_dir,
            arm=arm,
            model_dir=model_dir,
            prompt=prompt,
            n_tokens=n_tokens,
            stage="ladder",
        )
        rungs.append(rung)
        if rung["verdict"] == "PASS":
            highest_pass = n_tokens
        else:
            lowest_nonpass = n_tokens
            break

    bisect_steps: list[dict[str, Any]] = []
    if lowest_nonpass is not None:
        low, high = highest_pass, lowest_nonpass
        while high - low > resolution:
            midpoint = ((low + high) // 2 // round_to) * round_to
            if midpoint <= low or midpoint >= high:
                break
            prompt = prompt_for(
                root=root,
                tokenizer=tokenizer,
                n_tokens=midpoint,
                unit=unit,
                cache=prompt_cache,
            )
            rung = run_rung(
                root=root,
                cfg=cfg,
                p_cpus=p_cpus,
                work_dir=work_dir,
                run_dir=run_dir,
                arm=arm,
                model_dir=model_dir,
                prompt=prompt,
                n_tokens=midpoint,
                stage="bisect",
            )
            rungs.append(rung)
            bisect_steps.append({"n_tokens": midpoint, "verdict": rung["verdict"]})
            if rung["verdict"] == "PASS":
                low = midpoint
            else:
                high = midpoint
        highest_pass, lowest_nonpass = low, high

    reservations = [
        entry for rung in rungs for entry in rung["standing_reservation"] if isinstance(entry, dict)
    ]
    boundary = next((r for r in rungs if r["n_tokens"] == lowest_nonpass), None)

    summary = {
        "arm_id": arm["id"],
        "label": arm["label"],
        "load_sequence": arm["load_sequence"],
        "generate_device": arm["generate_device"],
        "started_utc": started,
        "finished_utc": _utc(),
        "ceiling_tokens": highest_pass,
        "ceiling_bracket": [highest_pass, lowest_nonpass],
        "ceiling_bracket_width": (
            None if lowest_nonpass is None else lowest_nonpass - highest_pass
        ),
        "ceiling_is_lower_bound": lowest_nonpass is None,
        "boundary_failure_modes": (boundary or {}).get("failure_modes", []),
        "boundary_repeat_spread": (
            None if boundary is None else f"{boundary['n_pass']}/{boundary['repeats']} passed"
        ),
        "rungs": rungs,
        "bisect_steps": bisect_steps,
        "standing_reservation_mb": _reservation_stats(reservations),
        "controls_at_start": controls_at_start,
        "memory_at_start": memory_at_start,
    }
    _log(
        "delta_n.arm_done",
        arm=arm["id"],
        ceiling=highest_pass,
        bracket=[highest_pass, lowest_nonpass],
    )
    return summary


def _reservation_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Median and range of the standing reservation, on both accounting bases."""

    def stat(key: str) -> dict[str, Any]:
        values = [float(e[key]) for e in entries if e.get(key) is not None]
        if not values:
            return {"median": None, "min": None, "max": None, "n": 0}
        return {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    return {
        "rss_delta_mb": stat("rss_delta_mb"),
        "available_delta_mb": stat("available_delta_mb"),
        "basis": (
            "process baseline (pre-import) to all backends resident, before any measured "
            "generation. The two bases disagree when weights are file-backed or shared; both "
            "are reported and neither is attributed here."
        ),
    }


# ------------------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------------------


def preflight(*, root: Path, model_dir: str, work_dir: Path) -> dict[str, Any]:
    """Device enumeration plus the per-model iGPU smoke test openvino#34390 requires."""
    import openvino as ov

    core = ov.Core()
    devices = list(core.available_devices)
    detail: dict[str, Any] = {"available_devices": devices}
    for device in devices:
        try:
            detail[f"{device}_full_name"] = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception as exc:
            detail[f"{device}_full_name"] = f"unavailable: {type(exc).__name__}"

    if "GPU" not in devices:
        return {
            "status": "UNSUPPORTED",
            "reason": "no GPU device exposed by OpenVINO; arm B cannot be constructed",
            "detail": detail,
        }

    smoke = run_child(
        root=root,
        work_dir=work_dir,
        spec={
            "model_dir": model_dir,
            "load_sequence": [{"device": "GPU", "properties": {}}],
            "generate_device": "GPU",
            "prompt_path": str(work_dir / "smoke_prompt.txt"),
            "max_new_tokens": 4,
            "affinity_cpus": None,
        },
        timeout_s=600.0,
        tag="preflight.igpu_smoke",
    )
    if smoke["outcome"] != "pass":
        return {
            "status": "UNSUPPORTED",
            "reason": (
                "iGPU smoke test failed; on a capability frontier an unsupported target is the "
                f"finding, not a retry: {smoke['failure_mode']}"
            ),
            "detail": detail,
            "smoke": smoke,
            "known_defect": "openvino#34390 CL_INVALID_WORK_GROUP_SIZE on Panther Lake iGPU",
        }
    return {"status": "SUPPORTED", "detail": detail, "smoke": smoke}


# ------------------------------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------------------------------


def build_verdict(arms: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    materiality = cfg["materiality"]
    ceiling_a = arms["A"]["ceiling_tokens"]
    ceiling_b = arms["B"]["ceiling_tokens"]
    ceiling_ap = arms["A_prime"]["ceiling_tokens"]

    delta_n = ceiling_a - ceiling_b
    aa_spread = abs(ceiling_a - ceiling_ap)
    fraction = float(materiality["aa_separability_fraction"])
    separable = aa_spread <= fraction * abs(delta_n) if delta_n != 0 else False

    if not separable:
        verdict = "not separable from noise"
        reason = (
            f"A/A spread {aa_spread} tokens exceeds {fraction:g}x |ΔN| = "
            f"{fraction * abs(delta_n):g} tokens; reported null regardless of the point estimate"
        )
    elif delta_n >= int(materiality["material_tokens"]):
        verdict = "material"
        reason = f"ΔN {delta_n} >= {materiality['material_tokens']} tokens"
    elif delta_n < int(materiality["dead_tokens"]):
        verdict = "immaterial"
        reason = f"ΔN {delta_n} < {materiality['dead_tokens']} tokens"
    else:
        verdict = "inconclusive"
        reason = (
            f"ΔN {delta_n} falls between {materiality['dead_tokens']} and "
            f"{materiality['material_tokens']} tokens"
        )

    return {
        "ceiling_tokens": {"A": ceiling_a, "B": ceiling_b, "A_prime": ceiling_ap},
        "ceiling_bracket": {key: arms[key]["ceiling_bracket"] for key in ("A", "B", "A_prime")},
        "boundary_repeat_spread": {
            key: arms[key]["boundary_repeat_spread"] for key in ("A", "B", "A_prime")
        },
        "delta_n_tokens": delta_n,
        "delta_n_definition": (
            "ceiling(A) - ceiling(B); positive means enabling the iGPU cost context"
        ),
        "aa_spread_tokens": aa_spread,
        "aa_separability_threshold_tokens": fraction * abs(delta_n),
        "separable_from_noise": separable,
        "verdict": verdict,
        "verdict_reason": reason,
        "materiality_prestated": dict(materiality),
        "resolution_note": (
            f"each ceiling is bracketed to {cfg['ladder']['bisect_resolution_tokens']} tokens; "
            "the bracket width bounds the resolution component of ΔN"
        ),
    }


# ------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------


def _assert_driver_matches_config(cfg: dict[str, Any]) -> None:
    """Refuse to start if the driver implements an older design than the config describes.

    ``configs/delta_n.yaml`` was revised to a phased, interleaved design; this driver still walks
    the arms sequentially and reads materiality keys the revised config no longer defines. Left
    unguarded that mismatch is at its worst: the ladder runs for hours, then :func:`build_verdict`
    raises a ``KeyError`` before anything is sealed, and the whole measurement is lost. Checking
    the keys up front turns a wasted afternoon into an immediate, legible stop.
    """
    missing = [
        key
        for key in ("aa_separability_fraction", "material_tokens", "dead_tokens")
        if key not in (cfg.get("materiality") or {})
    ]
    if missing:
        raise SeamError(
            "seam.tools.delta_n implements the sequential pre-phase design, but "
            f"configs/delta_n.yaml no longer defines materiality keys {missing}. The config now "
            "specifies four gated phases (preflight, acceptance, ceiling_a, delta_n), interleaved "
            "arms, and a working-set lock that this driver does not pass to the child. Running it "
            "would spend hours on the ladder and then fail before sealing. The driver must be "
            "brought up to the config before ΔN is launched; run the acceptance test first "
            "(seam.tools.fixed_throughput), which is complete and does match."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="harness validation: two tiny rungs, one repeat, no seal",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    resolved: ResolvedConfig = resolve_config(
        [root / _PLATFORM_PATH, root / _MEASUREMENT_PATH, root / _DELTA_N_PATH],
        repo_root=root,
    )
    cfg = resolved.data
    _assert_driver_matches_config(cfg)

    if not cfg["topology"]["verified"]:
        raise SeamError("topology is not verified; a cpu-p ceiling would be uninterpretable")
    p_cpus = [int(c) for c in cfg["topology"]["p_cpus"]]

    if args.smoke:
        cfg["ladder"] = {
            **cfg["ladder"],
            "rungs": [512, 1024],
            "repeats": 1,
            "bisect_resolution_tokens": 4096,
        }
        cfg["recovery"] = {**cfg["recovery"], "settle_s": 2, "max_wait_s": 30}

    model_spec_path = root / cfg["openvino"]["model_spec"]
    spec = load_local_spec(model_spec_path)
    model_dir = str(spec["ir_dir"])
    if not Path(model_dir).is_dir():
        raise SeamError(f"model IR directory missing: {model_dir}")

    run_id = str(uuid.uuid4())
    work_dir = root / "derived" / "delta_n" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "smoke_prompt.txt").write_text("Hello.", encoding="utf-8")

    _log("delta_n.start", run_id=run_id, smoke=args.smoke, model_dir=model_dir, **memory_now())

    pre = preflight(root=root, model_dir=model_dir, work_dir=work_dir)
    _log("delta_n.preflight", status=pre["status"], reason=pre.get("reason"))
    if pre["status"] != "SUPPORTED":
        raise SeamError(f"iGPU preflight {pre['status']}: {pre['reason']}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt_cache: dict[int, dict[str, Any]] = {}

    run_dir = open_in_progress_run(
        root=root,
        run_id=run_id,
        marker={
            "experiment_id": cfg["experiment_id"],
            "question": cfg["question"],
            "started_utc": _utc(),
            "arms": cfg["arm_order"],
            "smoke": args.smoke,
        },
    )
    run_dir.write_json("preflight.json", pre)

    arms_by_id = {arm["id"]: arm for arm in cfg["arms"]}
    arm_results: dict[str, dict[str, Any]] = {}
    for arm_id in cfg["arm_order"]:
        arm_results[arm_id] = run_arm(
            root=root,
            cfg=cfg,
            p_cpus=p_cpus,
            work_dir=work_dir,
            run_dir=run_dir,
            arm=arms_by_id[arm_id],
            model_dir=model_dir,
            tokenizer=tokenizer,
            prompt_cache=prompt_cache,
        )

    verdict = build_verdict(arm_results, cfg)
    _log("delta_n.verdict", **{k: v for k, v in verdict.items() if not isinstance(v, dict)})

    power = capture_power_state()
    summary = {
        "experiment_id": cfg["experiment_id"],
        "question": cfg["question"],
        "run_id": run_id,
        "smoke": args.smoke,
        "preflight": pre,
        "arm_order": cfg["arm_order"],
        "arms": arm_results,
        "verdict": verdict,
        "prompts": prompt_cache,
        "design": {
            "ladder": cfg["ladder"],
            "arms": cfg["arms"],
            "recovery": cfg["recovery"],
            "generation": cfg["generation"],
            "openvino": cfg["openvino"],
            "controls": cfg["controls"],
        },
        "scope_notes": {
            "not_measured": (
                "mechanism attribution (linear vs superlinear, activation vs KV), NPU, policy "
                "comparison, trajectories. No hypotheses are registered from this run."
            ),
            "npu_excluded_because": (
                "the NPU path requires NPUW_LLM_PREFILL_CHUNK_SIZE for openvino#34617, which "
                "bounds prefill activation memory by construction and would sit inside the "
                "comparison at the magnitude being detected"
            ),
            "confinement": (
                "not adopted on this platform (A1-A6 UNCLEAR, adopted_mechanism=null); CPU "
                "properties and process affinity are held identical across arms as a control"
            ),
            "paging_gate": (
                "reporting-only: recorded on every block, never used to discard a completion"
            ),
        },
        "power_start": asdict(power),
    }
    assert_acyclic(summary, label=f"delta_n summary run_id={run_id}")
    run_dir.write_json("delta_n_summary.json", summary)

    if args.smoke:
        _log("delta_n.smoke_complete", run_id=run_id, note="not sealed")
        print(json.dumps({"run_id": run_id, "sealed": False, "smoke": True}), flush=True)
        return 0

    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "microbench",
            "benchmark": "delta_n_context_ceiling_v1",
            "task_ids": [],
            "seed": 0,
            "n_repeats": int(cfg["ladder"]["repeats"]),
            "concurrency": 1,
        },
        condition_label=("delta_n|A=cpu-p|B=cpu-p+igpu|A_prime=AA|PAGING_GATE_REPORTING_ONLY"),
        repo_root=root,
        run_id=run_id,
        existing_run_dir=run_dir,
        allow_dirty=args.allow_dirty,
        summary=summary,
        model=manifest_model_block(
            spec=spec, spec_path=model_spec_path, reasoning_mode="thinking_off"
        ),
        drivers=asdict(runtime_info()),
        power_state=manifest_power_state(power, background_quiesced=False),
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
    )
    print(
        json.dumps(
            {
                "run_id": handle.run_id,
                "sealed": True,
                "verdict": verdict["verdict"],
                "delta_n_tokens": verdict["delta_n_tokens"],
                "aa_spread_tokens": verdict["aa_spread_tokens"],
                "ceilings": verdict["ceiling_tokens"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
