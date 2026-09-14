"""One fixed configuration, measured end to end. The instrument's own reproducibility test.

Two prior runs of the same configuration differed by 1.98x. Until that is explained or gone, no
difference this harness reports can be attributed to the thing that was varied, because a factor
of two is available from the machine for free. This tool exists to settle that: run it twice,
twenty minutes apart, on a machine in measurement mode, then compare the two sealed runs.

Nothing here is adjustable at the command line except which runs to compare. That is the point --
the prompt, the token counts, the CPU properties, the affinity, the working-set lock and the
repeat count all come from ``configs/delta_n.yaml`` and are recorded in the sealed manifest. Two
invocations differ in exactly one respect: when they happened.

The measured path is the one ΔN uses -- the same child process, the same TTFT instrument, the
same quiescence and canary envelope. A passing acceptance test therefore validates the instrument
that will take the real measurement rather than a simplified stand-in for it.

Reported:

``R_prefill``
    prompt tokens / TTFT. Sensitive to memory bandwidth and to anything competing for it.

``R_decode``
    (generated tokens - 1) / (wall - TTFT). The steady-state rate.

Both are reported per repeat and as a median, because the ratio between two invocations is only
interpretable against the spread within each of them.

The acceptance bound is derived from the runs, never asserted. ``s`` is the larger of the two
runs' within-run relative spread (CV) on the basis metric, the tolerance is twice it, and a pair
passes only if the between-run ratio falls inside that band *and* ``s`` is itself small enough to
mean anything. Baseline first, threshold from the baseline, both sealed -- the same discipline as
the canary and paging gates. ``--compare`` seals its own run, because ``s``, the band and the
verdict are numbers that exist in neither source run alone.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from seam.analysis.slice_stats import coefficient_of_variation
from seam.backends.local_openvino import runtime_info
from seam.config import ResolvedConfig, resolve_config
from seam.errors import SeamError
from seam.isolation import (
    assert_poolable,
    harness_termination_record,
    resolve_isolation_mode,
)
from seam.launch_context import assert_same_launch_context, resolve_launch_context
from seam.manifest import emit
from seam.model_provenance import load_local_spec, manifest_model_block
from seam.powerstate import capture_power_state, manifest_power_state
from seam.stdio_utf8 import configure_utf8_stdio
from seam.tools._winpower import assert_system_required
from seam.tools.acceptance_instrumentation import (
    capture_arm_memory,
    capture_uptime_wake,
    classify_arm_confinement,
    collect_kernel_power_events,
    correlate_kernel_power_with_repeats,
    host_memory_snapshot,
    memory_dispersion,
    reachability_sampler,
    summarize_power_gap,
)
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

BENCHMARK = "fixed_throughput_acceptance_v1"
VERDICT_BENCHMARK = "fixed_throughput_acceptance_verdict_v1"
ORCHESTRATED_BENCHMARK = "fixed_throughput_acceptance_orchestrated_v1"


def _stats(values: list[float | None]) -> dict[str, Any]:
    """Median, mean and within-run relative spread over the repeats that produced a value.

    ``cv`` is the quantity the acceptance band is derived from, so it is sealed per run rather
    than recomputed later from numbers that are no longer available.

    ``n_missing`` is reported rather than silently dropped: a repeat that completed without
    yielding a rate means the TTFT instrument failed on it, which is itself a reason not to trust
    the run.
    """
    present = [float(v) for v in values if v is not None]
    if not present:
        return {
            "median": None,
            "mean": None,
            "sd": None,
            "min": None,
            "max": None,
            "spread_ratio": None,
            "cv": None,
            "n": 0,
            "n_missing": len(values),
        }
    lo, hi = min(present), max(present)
    cv = coefficient_of_variation(present) if len(present) > 1 else math.nan
    return {
        "median": statistics.median(present),
        "mean": statistics.fmean(present),
        "sd": statistics.stdev(present) if len(present) > 1 else None,
        "min": lo,
        "max": hi,
        "spread_ratio": (hi / lo) if lo > 0 else None,
        # None rather than nan: nan does not survive JSON round-trip as a number, and an absent
        # spread must read as absent when the sealed summary is reloaded to reconstruct the band.
        "cv": None if math.isnan(cv) else cv,
        "n": len(present),
        "n_missing": len(values) - len(present),
    }


def _rate(record: dict[str, Any], key: str) -> float | None:
    generation = (record.get("result", {}).get("child") or {}).get("generation") or {}
    value = generation.get(key)
    return None if value is None else float(value)


def _sample_util_series(*, duration_s: float, interval_s: float = 0.1) -> list[list[float]]:
    """Collect per-core utilization samples for ``duration_s`` seconds."""
    import psutil

    psutil.cpu_percent(percpu=True)  # prime; first reading is meaningless
    samples: list[list[float]] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        samples.append([float(v) for v in psutil.cpu_percent(interval=None, percpu=True)])
        time.sleep(interval_s)
    return samples


class _UtilSampler:
    """Background per-core util sampler spanning one measured_repeat."""

    def __init__(self, *, interval_s: float = 0.1) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[list[float]] = []

    def __enter__(self) -> _UtilSampler:
        import psutil

        psutil.cpu_percent(percpu=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="acc-util", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        import psutil

        while not self._stop.is_set():
            self.samples.append([float(v) for v in psutil.cpu_percent(interval=None, percpu=True)])
            self._stop.wait(self._interval_s)


# ------------------------------------------------------------------------------------------
# Measure
# ------------------------------------------------------------------------------------------


def run_acceptance(
    *,
    allow_dirty: bool,
    arm_label: str = "arm",
    isolation_discipline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    resolved: ResolvedConfig = resolve_config(
        [root / _PLATFORM_PATH, root / _MEASUREMENT_PATH, root / _DELTA_N_PATH],
        repo_root=root,
    )
    cfg = resolved.data

    if not cfg["topology"]["verified"]:
        raise SeamError("topology is not verified; a cpu-p throughput number would not be citable")
    p_cpus = [int(c) for c in cfg["topology"]["p_cpus"]]
    lpe_cpus = [int(c) for c in cfg["topology"]["lpe_cpus"]]

    # Acceptance is the mode ΔN will use: declaration required, contradiction refused.
    launch_context, placement = resolve_launch_context(required=True)
    resolve_isolation_mode()  # fail closed before opening a run directory

    acceptance = cfg["acceptance"]
    context_tokens = int(acceptance["context_tokens"])
    max_new_tokens = int(acceptance["max_new_tokens"])
    repeats = int(acceptance["repeats"])
    # Harness never kills interactive software (refuse-only). Operator kills are unobservable.
    discipline = dict(isolation_discipline or {})
    discipline.setdefault("interactive_process_termination", harness_termination_record())
    discipline.setdefault(
        "pre_run_settle",
        {
            "configured_s": float((cfg.get("isolation") or {}).get("pre_run_settle_s") or 0),
            "actual_wait_s": None,
            "applied_on_this_path": False,
            "note": (
                "Authoritative pre_run_settle wait runs once in orchestrate() after the "
                "isolation check and before arm 1. Standalone (non-orchestrated) arms do not "
                "repeat that wait."
            ),
        },
    )

    model_spec_path = root / cfg["openvino"]["model_spec"]
    spec = load_local_spec(model_spec_path)
    model_dir = str(spec["ir_dir"])
    if not Path(model_dir).is_dir():
        raise SeamError(f"model IR directory missing: {model_dir}")

    run_id = str(uuid.uuid4())
    work_dir = root / "derived" / "fixed_throughput" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    uptime_wake = capture_uptime_wake()
    host_mem_start = host_memory_snapshot()

    _log(
        "fixed_throughput.start",
        run_id=run_id,
        arm_label=arm_label,
        launch_context=launch_context,
        session_id=placement.get("session_id"),
        window_station=placement.get("window_station"),
        context_tokens=context_tokens,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        **memory_now(),
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt = prompt_for(
        root=root,
        tokenizer=tokenizer,
        n_tokens=context_tokens,
        unit=str(cfg["ladder"]["filler_unit"]),
        cache={},
    )

    # The lock condition is the one ΔN will run under. configs/delta_n.yaml also lists
    # acceptance.working_set_lock_conditions: [off, request] -- that is a two-cell comparison of
    # the lock's effect, which is a different question and would make this run two configurations
    # rather than one. It is recorded here as deliberately not exercised.
    wslock_mode = str(cfg["working_set_lock"]["mode"])
    wslock = {
        "mode": wslock_mode,
        "minimum_bytes": int(cfg["working_set_lock"]["minimum_bytes"]),
        "maximum_bytes": int(cfg["working_set_lock"]["maximum_bytes"]),
    }

    child_spec = {
        "model_dir": model_dir,
        "load_sequence": [{"device": "CPU", "properties": dict(cfg["openvino"]["cpu_properties"])}],
        "generate_device": "CPU",
        "prompt_path": prompt["path"],
        "max_new_tokens": max_new_tokens,
        "affinity_cpus": list(p_cpus) if cfg["openvino"]["affinity"] == "p_cpus" else None,
        "rss_interval_s": 0.05,
        "wslock": wslock,
    }

    run_dir = open_in_progress_run(
        root=root,
        run_id=run_id,
        marker={
            "experiment_id": "fixed_throughput_acceptance",
            "purpose": "instrument reproducibility; 1.98x is the number to beat",
            "started_utc": _utc(),
            "context_tokens": context_tokens,
            "repeats": repeats,
            "arm_label": arm_label,
            "launch_context": launch_context,
        },
    )

    baseline_util = _sample_util_series(duration_s=2.0)
    records: list[dict[str, Any]] = []
    util_series: list[list[float]] = []
    for repeat in range(repeats):
        # Per-repeat AC/charging and wall-clock bounds - arm-level power_start alone cannot
        # locate a charge-controller flip or a Kernel-Power event onto one outlier.
        repeat_started_utc = _utc()
        power_at_repeat_start = asdict(capture_power_state())
        with _UtilSampler() as util:
            record = measured_repeat(
                root=root,
                cfg=cfg,
                p_cpus=p_cpus,
                work_dir=work_dir,
                child_spec=child_spec,
                label=f"fixed-throughput/{context_tokens}/{repeat}",
                tag=f"acc.n{context_tokens}.r{repeat}",
            )
        util_series.extend(util.samples)
        repeat_ended_utc = _utc()
        power_at_repeat_end = asdict(capture_power_state())
        record["repeat_index"] = repeat
        # Prefer the admitted attempt window from measured_repeat (excludes prior retries).
        env = record.get("envelope") or {}
        if not record.get("started_utc"):
            record["started_utc"] = repeat_started_utc
        if not record.get("ended_utc"):
            record["ended_utc"] = repeat_ended_utc
        if not record.get("window_utc"):
            record["window_utc"] = {
                "start": env.get("lock_acquired_utc") or record["started_utc"],
                "end": env.get("lock_released_utc") or record["ended_utc"],
                "source": (
                    "envelope.lock_acquired/released_utc"
                    if env.get("lock_acquired_utc") and env.get("lock_released_utc")
                    else "harness_repeat_wall_clock"
                ),
            }
        record["power_state_start"] = power_at_repeat_start
        record["power_state_end"] = power_at_repeat_end
        record["controls"] = observe_processes(cfg["controls"]["observe_processes"])
        record["power_request"] = (record.get("result") or {}).get("child", {}).get("power_request")
        records.append(record)
        run_dir.append_ndjson("repeats.ndjson", record)
        _log(
            "fixed_throughput.repeat",
            repeat=repeat,
            outcome=record["result"]["outcome"],
            r_prefill=_rate(record, "r_prefill_tok_s"),
            r_decode=_rate(record, "r_decode_tok_s"),
            started_utc=record["window_utc"]["start"],
            ended_utc=record["window_utc"]["end"],
            charging_start=power_at_repeat_start.get("charging"),
            battery_pct_start=power_at_repeat_start.get("battery_pct"),
        )

    failed = [r for r in records if r["result"]["outcome"] != "pass"]
    if failed:
        raise SeamError(
            f"{len(failed)}/{repeats} acceptance repeats did not complete "
            f"({sorted({r['result']['failure_mode'] for r in failed})}). A fixed configuration "
            "that cannot complete is not a throughput measurement. Stop and report."
        )

    # CV / median / spread use only admissible repeats. measured_repeat already retries
    # modern_standby_in_window (and other refusals) under admissibility.max_block_retries;
    # discarded attempts live on each record's attempts list and are never folded in here.
    admissible_records = [r for r in records if r.get("admissible", True)]
    if len(admissible_records) != len(records):
        raise SeamError(
            "fixed_throughput records contain inadmissible repeats after measured_repeat; "
            "that is a harness defect. Stop and report."
        )
    discarded_repeats = [
        attempt
        for r in records
        for attempt in (r.get("attempts") or [])
        if attempt.get("admissible") is False
    ]

    r_prefill = _stats([_rate(r, "r_prefill_tok_s") for r in admissible_records])
    r_decode = _stats([_rate(r, "r_decode_tok_s") for r in admissible_records])

    # This run's half of the derived band. Sealed here so the bound can be reconstructed from the
    # two runs alone, without re-deriving it from repeat values that live only in repeats.ndjson.
    basis_stats = {"r_prefill_tok_s": r_prefill, "r_decode_tok_s": r_decode}[
        str(acceptance["basis"])
    ]
    summary_spread = basis_stats["cv"]

    power = capture_power_state()
    memory_metrics = capture_arm_memory(records)
    memory_metrics["host_at_arm_start"] = host_mem_start
    confinement = classify_arm_confinement(
        p_cpus=p_cpus,
        lpe_cpus=lpe_cpus,
        util_series=util_series,
        baseline_series=baseline_util,
    )
    summary = {
        "experiment_id": "fixed_throughput_acceptance",
        "benchmark": BENCHMARK,
        "run_id": run_id,
        "arm_label": arm_label,
        "purpose": (
            "Measure one fixed configuration so that two invocations, separated in time, bound "
            "the instrument's between-run reproducibility. 1.98x is the prior unexplained ratio."
        ),
        "fixed_config": {
            "target": "cpu-p",
            "arm_id": "A",
            "context_tokens": context_tokens,
            "max_new_tokens": max_new_tokens,
            "repeats": repeats,
            "prompt": prompt,
            "cpu_properties": dict(cfg["openvino"]["cpu_properties"]),
            "affinity_cpus": child_spec["affinity_cpus"],
            "working_set_lock": wslock,
            "model_dir": model_dir,
        },
        "r_prefill_tok_s": r_prefill,
        "r_decode_tok_s": r_decode,
        "per_repeat": [
            {
                "repeat": r["repeat_index"],
                "admissible": r.get("admissible", True),
                "started_utc": r.get("started_utc"),
                "ended_utc": r.get("ended_utc"),
                "window_utc": r.get("window_utc"),
                "power_state_start": r.get("power_state_start"),
                "power_state_end": r.get("power_state_end"),
                "power_request": r.get("power_request")
                or (r["result"]["child"].get("power_request")),
                "r_prefill_tok_s": _rate(r, "r_prefill_tok_s"),
                "r_decode_tok_s": _rate(r, "r_decode_tok_s"),
                "wall_s": (r["result"]["child"].get("generation") or {}).get("wall_s"),
                "ttft_source": (r["result"]["child"].get("generation") or {}).get("ttft_source"),
                "prompt_tokens_reported": (
                    (r["result"]["child"].get("generation") or {}).get("prompt_tokens_reported")
                ),
                "completion_tokens_reported": (
                    (r["result"]["child"].get("generation") or {}).get("completion_tokens_reported")
                ),
                "peak_rss_bytes": (r["result"]["child"].get("generation") or {}).get(
                    "peak_rss_bytes"
                ),
                "peak_commit_bytes": (r["result"]["child"].get("generation") or {}).get(
                    "peak_commit_bytes"
                ),
                "attempt_used": r["attempt_used"],
                "inadmissible_attempts": [a["reason"] for a in r["attempts"]],
                "discarded_attempts": list(r.get("attempts") or []),
                "envelope": r["envelope"],
            }
            for r in records
        ],
        "discarded_repeats": discarded_repeats,
        "admissibility": {
            "max_block_retries": int(cfg["admissibility"]["max_block_retries"]),
            "reject_kernel_power_ids": list(
                cfg["admissibility"].get("reject_kernel_power_ids") or []
            ),
            "modern_standby_inadmissible": bool(
                cfg["admissibility"].get("modern_standby_inadmissible", True)
            ),
            "rule": (
                "Any repeat whose [started_utc, ended_utc] window contains a Kernel-Power "
                "event whose Id is in reject_kernel_power_ids is INADMISSIBLE "
                "(reason=modern_standby_in_window), retried under max_block_retries, "
                "recorded verbatim with its events, and excluded from CV/median/spread."
            ),
            "n_discarded": len(discarded_repeats),
            "stats_basis": "admissible_repeats_only",
        },
        "power_request": {
            "child_role": "measurement_child",
            "mechanism": "PowerCreateRequest/PowerSetRequest",
            "request_type": "PowerRequestSystemRequired",
            "per_repeat": [
                r.get("power_request") or (r["result"]["child"].get("power_request"))
                for r in records
            ],
            "note": (
                "Held in the measurement child for the life of each generate. Not a "
                "power-policy change; idle timeout is untouched."
            ),
        },
        "memory_metrics": memory_metrics,
        "confinement_classification": confinement,
        "uptime_wake": uptime_wake,
        "acceptance_criterion": {
            "gate_mode": "derived_from_within_run_spread",
            "basis": str(acceptance["basis"]),
            "spread_estimator": str(acceptance["spread_estimator"]),
            "band_multiple": float(acceptance["band_multiple"]),
            "max_within_run_spread": float(acceptance["max_within_run_spread"]),
            "rule": (
                "s = max over the two runs of the within-run relative spread on `basis`; "
                "band = band_multiple * s; PASS iff |ratio - 1| <= band AND "
                "s <= max_within_run_spread"
            ),
            "prior_unexplained_ratio": float(acceptance["prior_unexplained_ratio"]),
            "gap_s": int(acceptance["gap_s"]),
        },
        # Per-run observation, not part of the criterion. Older sealed summaries nested this
        # under acceptance_criterion; compare() ignores it either way (rule fields only).
        "this_run_within_run_spread": {
            "basis": str(acceptance["basis"]),
            "cv": summary_spread,
        },
        "not_exercised": {
            "working_set_lock_conditions": (
                "configs/delta_n.yaml lists [off, request]. Only the mode ΔN will use is measured "
                f"here ({wslock_mode}); sweeping both would make this two configurations rather "
                "than one, which is not what an acceptance test is."
            )
        },
        "isolation_discipline": discipline,
        "power_start": asdict(power),
    }
    assert_acyclic(summary, label=f"fixed_throughput summary run_id={run_id}")

    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            # An A/A test in the blueprint's sense: one configuration measured twice so the noise
            # floor is known before any comparison is trusted. Recording it as `microbench` would
            # disconnect it from the gate it exists to satisfy.
            "kind": "aa",
            "benchmark": BENCHMARK,
            "task_ids": [],
            "seed": 0,
            "n_repeats": repeats,
            "concurrency": 1,
        },
        condition_label=f"fixed_throughput|cpu-p|n={context_tokens}|wslock={wslock_mode}",
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

    return {
        "run_id": handle.run_id,
        "sealed": True,
        "isolation_mode": handle.manifest["isolation_mode"],
        "launch_context": handle.manifest.get("launch_context"),
        "session_id": handle.manifest.get("session_id"),
        "window_station": handle.manifest.get("window_station"),
        "arm_label": arm_label,
        "context_tokens": context_tokens,
        "R_prefill_tok_s": r_prefill["median"],
        "R_decode_tok_s": r_decode["median"],
        "within_run_spread_cv": {
            "basis": str(acceptance["basis"]),
            "value": summary_spread,
            "R_prefill": r_prefill["cv"],
            "R_decode": r_decode["cv"],
        },
        "memory_metrics": memory_metrics,
        "confinement_classification": {
            "verdict": confinement["verdict"],
            "core_states": confinement.get("core_states"),
        },
        "uptime_wake": uptime_wake,
        "prompt_sha256": prompt["sha256"],
        "config_hash": handle.manifest["config_hash"],
        "isolation_discipline": discipline,
    }


# ------------------------------------------------------------------------------------------
# Compare
# ------------------------------------------------------------------------------------------


def _load_run(root: Path, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run_path = root / "raw" / run_id
    manifest_path = run_path / "manifest.json"
    summary_path = run_path / "summary.json"
    if not manifest_path.is_file():
        raise SeamError(f"no sealed manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("benchmark") != BENCHMARK:
        raise SeamError(
            f"run {run_id} is not a {BENCHMARK} run (benchmark={summary.get('benchmark')!r})"
        )
    return manifest, summary


# Rule fields of acceptance_criterion. Observations such as this_run_within_run_spread are
# excluded: they describe one run and must not make two otherwise-identical criteria unequal.
_CRITERION_RULE_KEYS = (
    "basis",
    "spread_estimator",
    "band_multiple",
    "max_within_run_spread",
    "gate_mode",
    "gap_s",
    "rule",
)


def criterion_rule(criterion: dict[str, Any]) -> dict[str, Any]:
    """Project an acceptance_criterion dict onto the fields that define the gate rule."""
    missing = [k for k in _CRITERION_RULE_KEYS if k not in criterion]
    if missing:
        raise SeamError("acceptance_criterion is missing rule field(s): " + ", ".join(missing))
    return {k: criterion[k] for k in _CRITERION_RULE_KEYS}


def derive_verdict(
    *,
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    criterion: dict[str, Any],
) -> dict[str, Any]:
    """Derive the acceptance band from the two runs and apply it.

    The band is a property of the measurement, not a constant: ``s`` is the larger of the two
    runs' within-run relative spread on the basis metric, and the tolerance is ``band_multiple``
    times it. An instrument is asked to reproduce itself to within the noise it demonstrably has,
    which is the same shape as the canary and paging gates -- baseline first, threshold from the
    baseline, both recorded.

    The second condition is what stops the first from being self-satisfying. ``|ratio - 1| <= 2s``
    can always be met by a large enough ``s``, so without a ceiling on the spread a pair of runs
    could pass on the strength of its own noise.
    """
    basis = str(criterion["basis"])
    band_multiple = float(criterion["band_multiple"])
    max_spread = float(criterion["max_within_run_spread"])

    median_a = summary_a[basis]["median"]
    median_b = summary_b[basis]["median"]
    cv_a = summary_a[basis]["cv"]
    cv_b = summary_b[basis]["cv"]

    reasons: list[str] = []
    # `ratio` is max/min, so it is >= 1 and orientation-free -- the convention 1.98x was quoted
    # in. The signed ratio is kept alongside it because which of the two runs was faster says
    # whether the machine drifted up or down, which max/min discards.
    if median_a is None or median_b is None or min(median_a, median_b) <= 0:
        ratio = None
        signed_ratio = None
        reasons.append(f"{basis} median missing or non-positive in one or both runs")
    else:
        ratio = max(median_a, median_b) / min(median_a, median_b)
        signed_ratio = median_b / median_a

    if cv_a is None or cv_b is None:
        spread = None
        reasons.append(
            "within-run spread could not be computed for one or both runs (fewer than two "
            "repeats produced a rate); the band is not derivable and no pass is available"
        )
    else:
        spread = max(float(cv_a), float(cv_b))

    band = None if spread is None else band_multiple * spread
    within_band = None if (ratio is None or band is None) else abs(ratio - 1.0) <= band
    spread_acceptable = None if spread is None else spread <= max_spread

    if ratio is not None and band is not None and not within_band:
        reasons.append(
            f"|ratio - 1| = {abs(ratio - 1.0):.4f} exceeds the derived band "
            f"{band_multiple:g} x s = {band:.4f}"
        )
    if spread is not None and not spread_acceptable:
        reasons.append(
            f"within-run spread s = {spread:.4f} exceeds {max_spread:.2f}; a band derived from a "
            "run this noisy describes a machine that cannot support a claim, so the ratio is not "
            "evidence of reproducibility whatever its value"
        )

    if ratio is None or spread is None:
        verdict = "INDETERMINATE"
    elif not reasons:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "gate_mode": "derived_from_within_run_spread",
        "basis": basis,
        "spread_estimator": str(criterion["spread_estimator"]),
        "within_run_spread": {"run_a": cv_a, "run_b": cv_b},
        "s": spread,
        "s_source": (
            None if spread is None else ("run_a" if float(cv_a) >= float(cv_b) else "run_b")
        ),
        "band_multiple": band_multiple,
        "band": band,
        "max_within_run_spread": max_spread,
        "medians": {"run_a": median_a, "run_b": median_b},
        "ratio": ratio,
        "ratio_convention": "max(median)/min(median), so >= 1 regardless of run order",
        "signed_ratio_b_over_a": signed_ratio,
        "abs_ratio_minus_one": None if ratio is None else abs(ratio - 1.0),
        "within_band": within_band,
        "spread_acceptable": spread_acceptable,
        "verdict": verdict,
        "reasons": reasons,
        "prior_unexplained_ratio": float(criterion["prior_unexplained_ratio"]),
        "derivation": (
            "s = max(cv_a, cv_b) on `basis`; band = band_multiple * s; "
            "PASS iff |ratio - 1| <= band and s <= max_within_run_spread. Every input is present "
            "in this record, so the bound can be recomputed without the source runs."
        ),
    }


def compare(run_a: str, run_b: str, *, allow_dirty: bool = False) -> dict[str, Any]:
    """Derive the band from two sealed runs, apply it, and seal the verdict as its own run.

    The verdict is sealed rather than printed because ``s``, the band and the ratio are emitted
    numbers that exist in neither source run on its own. Without a manifest of their own they
    would be numbers with no ``run_id`` behind them.
    """
    root = Path(__file__).resolve().parents[2]
    manifest_a, summary_a = _load_run(root, run_a)
    manifest_b, summary_b = _load_run(root, run_b)

    # The rule this whole exercise exists to enforce. Checked before anything is computed, so a
    # cross-mode ratio is never even produced, let alone printed.
    isolation_mode = assert_poolable([manifest_a, manifest_b])
    compared_launch_context = assert_same_launch_context([manifest_a, manifest_b])

    mismatches: list[str] = []
    if manifest_a["config_hash"] != manifest_b["config_hash"]:
        mismatches.append(
            f"config_hash {manifest_a['config_hash'][:12]} != {manifest_b['config_hash'][:12]}"
        )
    sha_a = summary_a["fixed_config"]["prompt"]["sha256"]
    sha_b = summary_b["fixed_config"]["prompt"]["sha256"]
    if sha_a != sha_b:
        mismatches.append(f"prompt sha256 {sha_a[:12]} != {sha_b[:12]}")
    lock_a = summary_a["fixed_config"]["working_set_lock"]
    lock_b = summary_b["fixed_config"]["working_set_lock"]
    if lock_a != lock_b:
        mismatches.append(f"working_set_lock {lock_a} != {lock_b}")
    if mismatches:
        raise SeamError(
            "refusing to report a ratio between runs that are not the same configuration: "
            + "; ".join(mismatches)
        )

    # The criterion is read from the sealed run rather than from the current config, so a later
    # edit to configs/delta_n.yaml cannot retroactively change the bound a sealed pair was judged
    # against. A mismatch between the two runs' *rule* fields is itself disqualifying.
    # this_run_within_run_spread (present under acceptance_criterion in older seals, or at the
    # summary root in newer ones) is a per-run observation and is not part of the rule.
    criterion_a = summary_a["acceptance_criterion"]
    criterion_b = summary_b["acceptance_criterion"]
    if criterion_rule(criterion_a) != criterion_rule(criterion_b):
        raise SeamError(
            "the two runs were sealed under different acceptance criteria; the bound they would "
            "be judged against is ambiguous. Re-run both under one configuration."
        )
    # Carry prior_unexplained_ratio and other non-observation keys from run_a for the sealed
    # record; strip this_run_within_run_spread so the verdict's criterion is the rule only.
    criterion = {k: v for k, v in criterion_a.items() if k != "this_run_within_run_spread"}

    gate = derive_verdict(summary_a=summary_a, summary_b=summary_b, criterion=criterion)

    # Reported alongside the gated metric, never gating: a prefill ratio that disagrees with the
    # decode verdict is a signal about which part of the pipeline moved, and suppressing it would
    # discard the most useful diagnostic the acceptance test produces.
    secondary_basis = (
        "r_prefill_tok_s" if criterion["basis"] == "r_decode_tok_s" else "r_decode_tok_s"
    )
    secondary = derive_verdict(
        summary_a=summary_a,
        summary_b=summary_b,
        criterion={**criterion, "basis": secondary_basis},
    )

    mem_a = summary_a.get("memory_metrics") or {}
    mem_b = summary_b.get("memory_metrics") or {}
    conf_a = summary_a.get("confinement_classification") or {}
    conf_b = summary_b.get("confinement_classification") or {}

    result = {
        "run_a": run_a,
        "run_b": run_b,
        "compared_isolation_mode": isolation_mode,
        "compared_launch_context": compared_launch_context,
        "timestamps_utc": [manifest_a["timestamp_utc"], manifest_b["timestamp_utc"]],
        "R_prefill_tok_s": {
            "run_a": summary_a["r_prefill_tok_s"]["median"],
            "run_b": summary_b["r_prefill_tok_s"]["median"],
        },
        "R_decode_tok_s": {
            "run_a": summary_a["r_decode_tok_s"]["median"],
            "run_b": summary_b["r_decode_tok_s"]["median"],
        },
        "memory_dispersion": memory_dispersion(mem_a, mem_b) if mem_a and mem_b else None,
        "confinement_classifications": {
            "run_a": {"verdict": conf_a.get("verdict"), "core_states": conf_a.get("core_states")},
            "run_b": {"verdict": conf_b.get("verdict"), "core_states": conf_b.get("core_states")},
        },
        "instrument_gate": gate,
        "secondary_not_gating": secondary,
        "verdict": gate["verdict"],
    }

    root_config: ResolvedConfig = resolve_config(
        [root / _PLATFORM_PATH, root / _MEASUREMENT_PATH, root / _DELTA_N_PATH],
        repo_root=root,
    )
    verdict_run_id = str(uuid.uuid4())
    sealed_summary = {
        "experiment_id": "fixed_throughput_acceptance",
        "benchmark": VERDICT_BENCHMARK,
        "run_id": verdict_run_id,
        "purpose": (
            "Derive the acceptance band from two sealed fixed-config runs and apply it. Holds the "
            "numbers that exist in neither source run alone: s, the band, and the verdict."
        ),
        "source_runs": {
            "run_a": {
                "run_id": run_a,
                "timestamp_utc": manifest_a["timestamp_utc"],
                "config_hash": manifest_a["config_hash"],
                "isolation_mode": manifest_a["isolation_mode"],
                "launch_context": manifest_a.get("launch_context"),
                "session_id": manifest_a.get("session_id"),
                "window_station": manifest_a.get("window_station"),
                "r_prefill_tok_s": summary_a["r_prefill_tok_s"],
                "r_decode_tok_s": summary_a["r_decode_tok_s"],
                "memory_metrics": mem_a,
                "confinement_classification": conf_a,
            },
            "run_b": {
                "run_id": run_b,
                "timestamp_utc": manifest_b["timestamp_utc"],
                "config_hash": manifest_b["config_hash"],
                "isolation_mode": manifest_b["isolation_mode"],
                "launch_context": manifest_b.get("launch_context"),
                "session_id": manifest_b.get("session_id"),
                "window_station": manifest_b.get("window_station"),
                "r_prefill_tok_s": summary_b["r_prefill_tok_s"],
                "r_decode_tok_s": summary_b["r_decode_tok_s"],
                "memory_metrics": mem_b,
                "confinement_classification": conf_b,
            },
        },
        "compared_isolation_mode": isolation_mode,
        "compared_launch_context": compared_launch_context,
        "acceptance_criterion": criterion,
        "instrument_gate": gate,
        "secondary_not_gating": secondary,
        "memory_dispersion": result["memory_dispersion"],
        "confinement_classifications": result["confinement_classifications"],
        "fixed_config": summary_a["fixed_config"],
        "isolation_mode_note": (
            "This run's own isolation_mode describes the process that computed the verdict, which "
            "measures nothing. The mode that bears on validity is compared_isolation_mode, the "
            "single mode both source runs were measured under."
        ),
    }
    assert_acyclic(sealed_summary, label=f"acceptance verdict run_id={verdict_run_id}")

    handle = emit(
        config=root_config,
        target="cpu-p",
        workload={
            "kind": "aa",
            "benchmark": VERDICT_BENCHMARK,
            "task_ids": [run_a, run_b],
            "seed": 0,
            "n_repeats": 2,
            "concurrency": 1,
        },
        condition_label=f"acceptance_verdict|{run_a}|{run_b}",
        repo_root=root,
        run_id=verdict_run_id,
        allow_dirty=allow_dirty,
        summary=sealed_summary,
        drivers=asdict(runtime_info()),
        thermal={"regime": "confound", "excluded": False},
        self_check="pass" if gate["verdict"] != "INDETERMINATE" else "fail",
        require_launch_context=False,
    )
    result["verdict_run_id"] = handle.run_id
    return result


# ------------------------------------------------------------------------------------------
# Orchestrate (arm 1 → gap → arm 2 → sealed verdict)
# ------------------------------------------------------------------------------------------


def _wait_pre_run_settle(configured_s: float) -> dict[str, Any]:
    """Idle inside the detached orchestrator after isolation passes and before arm 1.

    Distinct from ``recovery.settle_s`` (post-child reclaim). Recorded with the observed wait so
    a cut-short sleep cannot be mistaken for the configured settle.
    """
    started_utc = _utc()
    t0 = time.monotonic()
    _log(
        "fixed_throughput.pre_run_settle_start",
        configured_s=configured_s,
        started_utc=started_utc,
        distinct_from="recovery.settle_s",
    )
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
        "note": (
            "Wait for memory reclaim/writeback after the operator closed interactive software. "
            "The harness refuses while contending processes are resident; it does not kill them."
        ),
    }
    _log(
        "fixed_throughput.pre_run_settle_end",
        configured_s=configured_s,
        actual_wait_s=actual_wait_s,
        ended_utc=ended_utc,
    )
    return record


def orchestrate(*, allow_dirty: bool) -> dict[str, Any]:
    """One detached job: settle, seal arm 1, idle ``gap_s``, seal arm 2, seal the verdict.

    This is the path ΔN's acceptance gate uses. Launch context must be ``ssh_detached`` and
    isolation_mode ``remote``. Cursor (and other tier-1 CONTENDING processes) must already be
    closed - ``resolve_isolation_mode`` refuses otherwise (tier-2 shell/vendor agents are
    recorded only). After that check passes, the orchestrator idles
    ``isolation.pre_run_settle_s`` *inside this process* before arm 1 so closing SSH cannot cut
    the settle short.
    """
    root = Path(__file__).resolve().parents[2]
    resolved: ResolvedConfig = resolve_config(
        [root / _PLATFORM_PATH, root / _MEASUREMENT_PATH, root / _DELTA_N_PATH],
        repo_root=root,
    )
    cfg = resolved.data
    gap_s = int(cfg["acceptance"]["gap_s"])
    isolation_block = cfg.get("isolation") or {}
    if "pre_run_settle_s" not in isolation_block:
        raise SeamError(
            "configs/delta_n.yaml must declare isolation.pre_run_settle_s (distinct from "
            "recovery.settle_s). Refusing to orchestrate without a pre-registered settle."
        )
    pre_run_settle_s = float(isolation_block["pre_run_settle_s"])

    launch_context, placement = resolve_launch_context(required=True)
    isolation_mode, _evidence = resolve_isolation_mode()
    if launch_context != "ssh_detached":
        raise SeamError(
            f"orchestrated acceptance requires launch_context=ssh_detached, got {launch_context!r}. "
            "A Cursor-session or SSH-foreground run must not be labelled as the detached path."
        )
    if isolation_mode != "remote":
        raise SeamError(
            f"orchestrated acceptance requires isolation_mode=remote, got {isolation_mode!r}"
        )

    # Refuse-only: this harness never terminates interactive software. Record that explicitly
    # rather than inventing kill timestamps from process absence.
    termination = harness_termination_record()

    # Orchestrator holds PowerRequestSystemRequired across settle, arms, and gap so idle
    # between children cannot enter Modern Standby either. Children also assert independently.
    orch_power = assert_system_required(
        reason=(
            "SEAM fixed_throughput orchestrator; PowerRequestSystemRequired across "
            "pre_run_settle, arms, and gap"
        ),
        role="acceptance_orchestrator",
    )
    _log(
        "fixed_throughput.orchestrate_power_request",
        succeeded=orch_power.record.get("succeeded"),
        mechanism=orch_power.record.get("mechanism"),
        error=orch_power.record.get("error"),
    )

    try:
        pre_run_settle = _wait_pre_run_settle(pre_run_settle_s)
        arm_discipline = {
            "pre_run_settle": pre_run_settle,
            "interactive_process_termination": termination,
            "power_request": dict(orch_power.record),
        }

        window_start = _utc()
        reach_path = (
            root
            / "derived"
            / "fixed_throughput"
            / "_orchestrate"
            / f"reach_{window_start.replace(':', '')}.json"
        )
        reach = reachability_sampler(host="1.1.1.1", interval_s=30.0, log_path=reach_path)
        reach.start()

        _log(
            "fixed_throughput.orchestrate_start",
            gap_s=gap_s,
            pre_run_settle_s=pre_run_settle_s,
            pre_run_settle_actual_wait_s=pre_run_settle["actual_wait_s"],
            launch_context=launch_context,
            session_id=placement.get("session_id"),
            window_station=placement.get("window_station"),
            isolation_mode=isolation_mode,
            power_request_succeeded=orch_power.record.get("succeeded"),
        )

        try:
            arm1 = run_acceptance(
                allow_dirty=allow_dirty,
                arm_label="arm_1",
                isolation_discipline=arm_discipline,
            )
            _log("fixed_throughput.orchestrate_gap", gap_s=gap_s, after_run_id=arm1["run_id"])
            time.sleep(gap_s)
            # Same discipline for arm 2: harness still has not killed anything; operator kill
            # time remains unobservable. pre_run_settle already completed before arm 1.
            arm2 = run_acceptance(
                allow_dirty=allow_dirty,
                arm_label="arm_2",
                isolation_discipline=arm_discipline,
            )
        finally:
            reachability = reach.stop()

        window_end = _utc()
        kernel_power = collect_kernel_power_events(start_utc=window_start, end_utc=window_end)
        power_gap = summarize_power_gap(
            arm1_uptime=arm1["uptime_wake"],
            arm2_uptime=arm2["uptime_wake"],
            kernel_power=kernel_power,
        )

        verdict = compare(arm1["run_id"], arm2["run_id"], allow_dirty=allow_dirty)

        # Per-repeat coincidence with Kernel-Power: the decisive confound test. Windows come from
        # each sealed arm's per_repeat records (lock envelope when present).
        def _repeat_rows(arm: dict[str, Any], arm_label: str) -> list[dict[str, Any]]:
            summary = arm.get("summary") or {}
            rows: list[dict[str, Any]] = []
            for pr in summary.get("per_repeat") or []:
                window = pr.get("window_utc") or {}
                env = pr.get("envelope") or {}
                start = window.get("start") or env.get("lock_acquired_utc") or pr.get("started_utc")
                end = window.get("end") or env.get("lock_released_utc") or pr.get("ended_utc")
                rows.append(
                    {
                        "arm": arm_label,
                        "run_id": arm.get("run_id"),
                        "repeat": pr.get("repeat"),
                        "start_utc": start,
                        "end_utc": end,
                        "window_source": window.get("source")
                        or (
                            "envelope.lock_acquired/released_utc"
                            if env.get("lock_acquired_utc") and env.get("lock_released_utc")
                            else "unknown"
                        ),
                        "r_decode_tok_s": pr.get("r_decode_tok_s"),
                        "r_prefill_tok_s": pr.get("r_prefill_tok_s"),
                        "wall_s": pr.get("wall_s"),
                        "power_state_start": pr.get("power_state_start"),
                        "power_state_end": pr.get("power_state_end"),
                    }
                )
            return rows

        # run_acceptance returns a compact handle; reload sealed summaries for per_repeat windows.
        def _load_arm_summary(run_id: str) -> dict[str, Any]:
            path = root / "raw" / run_id / "summary.json"
            return json.loads(path.read_text(encoding="utf-8"))

        arm1_full = {"run_id": arm1["run_id"], "summary": _load_arm_summary(arm1["run_id"])}
        arm2_full = {"run_id": arm2["run_id"], "summary": _load_arm_summary(arm2["run_id"])}
        repeat_kernel_power = correlate_kernel_power_with_repeats(
            events=list(kernel_power.get("events") or []),
            repeats=_repeat_rows(arm1_full, "arm_1") + _repeat_rows(arm2_full, "arm_2"),
        )

        # Seal an orchestrator summary that holds the gap instruments. The throughput verdict already
        # has its own run_id; this run cites both arms and the verdict so every additive number traces.
        orch_run_id = str(uuid.uuid4())
        orch_summary = {
            "experiment_id": "fixed_throughput_acceptance",
            "benchmark": ORCHESTRATED_BENCHMARK,
            "run_id": orch_run_id,
            "purpose": (
                "Orchestrated acceptance: pre_run_settle, arm1, gap_s idle, arm2, derived verdict. "
                "Holds the gap instruments (Kernel-Power, reachability) that span both arms."
            ),
            "gap_s": gap_s,
            "isolation_discipline": {
                "pre_run_settle": pre_run_settle,
                "interactive_process_termination": termination,
            },
            "power_request": dict(orch_power.record),
            "window_utc": {"start": window_start, "end": window_end},
            "launch_context": launch_context,
            "session_id": placement.get("session_id"),
            "window_station": placement.get("window_station"),
            "isolation_mode": isolation_mode,
            "arm_1": arm1,
            "arm_2": arm2,
            "verdict_run_id": verdict.get("verdict_run_id"),
            "instrument_gate": verdict.get("instrument_gate"),
            "memory_dispersion": verdict.get("memory_dispersion"),
            "confinement_classifications": verdict.get("confinement_classifications"),
            "power_gap": power_gap,
            "kernel_power": kernel_power,
            "kernel_power_events": list(kernel_power.get("events") or []),
            "repeat_kernel_power": repeat_kernel_power,
            "reachability": {
                "host": reachability.get("host"),
                "interval_s": reachability.get("interval_s"),
                "n_samples": reachability.get("n_samples"),
                "n_unreachable": reachability.get("n_unreachable"),
                "log_path": reachability.get("log_path"),
                # Full sample list lives in the log file; keep the seal lean.
                "samples_head": (reachability.get("samples") or [])[:3],
                "samples_tail": (reachability.get("samples") or [])[-3:],
            },
            "signed_ratio_arm2_over_arm1": (verdict.get("instrument_gate") or {}).get(
                "signed_ratio_b_over_a"
            ),
            "ratio_max_over_min": (verdict.get("instrument_gate") or {}).get("ratio"),
            "narrative_gate": (
                "AM-027(b) on Blueprint (narrative gate: every quoted number carries a run_id). "
                "AMENDMENTS-side AM-027 was tombstoned into AM-034; do not invent AM-034(b)."
            ),
        }
        assert_acyclic(orch_summary, label=f"orchestrated acceptance run_id={orch_run_id}")
        handle = emit(
            config=resolved,
            target="cpu-p",
            workload={
                "kind": "aa",
                "benchmark": ORCHESTRATED_BENCHMARK,
                "task_ids": [arm1["run_id"], arm2["run_id"], str(verdict.get("verdict_run_id"))],
                "seed": 0,
                "n_repeats": 2,
                "concurrency": 1,
            },
            condition_label=f"acceptance_orchestrated|{arm1['run_id']}|{arm2['run_id']}",
            repo_root=root,
            run_id=orch_run_id,
            allow_dirty=allow_dirty,
            summary=orch_summary,
            drivers=asdict(runtime_info()),
            thermal={"regime": "confound", "excluded": False},
            self_check="pass" if verdict.get("verdict") != "INDETERMINATE" else "fail",
            launch_context=launch_context,
            session_id=placement.get("session_id"),
            window_station=placement.get("window_station"),
            require_launch_context=True,
        )

        return {
            "orchestrator_run_id": handle.run_id,
            "arm_1_run_id": arm1["run_id"],
            "arm_2_run_id": arm2["run_id"],
            "verdict_run_id": verdict.get("verdict_run_id"),
            "verdict": verdict.get("verdict"),
            "signed_ratio_arm2_over_arm1": (verdict.get("instrument_gate") or {}).get(
                "signed_ratio_b_over_a"
            ),
            "ratio_max_over_min": (verdict.get("instrument_gate") or {}).get("ratio"),
            "instrument_gate": verdict.get("instrument_gate"),
            "memory_dispersion": verdict.get("memory_dispersion"),
            "confinement_classifications": verdict.get("confinement_classifications"),
            "power_gap": power_gap,
            "power_request": dict(orch_power.record),
            "kernel_power_events": list(kernel_power.get("events") or []),
            "repeat_kernel_power": repeat_kernel_power,
            "reachability_n_unreachable": reachability.get("n_unreachable"),
            "launch_context": launch_context,
            "session_id": placement.get("session_id"),
            "window_station": placement.get("window_station"),
            "isolation_mode": isolation_mode,
            "gap_s": gap_s,
            "isolation_discipline": {
                "pre_run_settle": pre_run_settle,
                "interactive_process_termination": termination,
            },
        }
    finally:
        orch_power.release()


def main(argv: list[str] | None = None) -> int:
    # Before argparse, which prints this module's docstring -- and the Δ in it is enough to raise
    # UnicodeEncodeError on a cp1252 console. Two completed runs have already been lost to
    # write-path encoding defects found only after the measurement finished (C2d / C9).
    configure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help="report the ratio between two sealed runs instead of measuring",
    )
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help=(
            "after isolation check, idle isolation.pre_run_settle_s, seal arm 1, idle "
            "acceptance.gap_s, seal arm 2, seal the derived verdict - one detached job. "
            "Requires SEAM_ISOLATION_MODE=remote and SEAM_LAUNCH_CONTEXT=ssh_detached."
        ),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help=(
            "also write the result here. A detached run has no console to return to, so the "
            "launcher needs a file it can find the outcome in."
        ),
    )
    args = parser.parse_args(argv)

    if args.compare and args.orchestrate:
        raise SystemExit("refusing --compare and --orchestrate together")

    if args.compare:
        result = compare(*args.compare, allow_dirty=args.allow_dirty)
    elif args.orchestrate:
        result = orchestrate(allow_dirty=args.allow_dirty)
    else:
        result = run_acceptance(allow_dirty=args.allow_dirty)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered, encoding="utf-8")
    print(rendered, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
