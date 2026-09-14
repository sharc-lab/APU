"""M2.1 - S1 battery-counter characterization (spec §7 M2.1, §3.1, §3.7).

Polls ``root\\WMI`` ``BatteryStatus`` at a configured rate (default 10 Hz), writes
``samples.ndjson``, and - after a measurement run - derives update period, quantization
step, settle_s, and SoC-window recommendations.

This is **not** the general §3.6 unified sampler. No RAPL, thermal, or PDH code lives here.

Measurement requires ``battery-pinned`` (on battery, not charging). The CLI refuses a
profile mismatch by emitting a citable refusal manifest, then stopping.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from seam.config import ResolvedConfig, load_platform_config
from seam.errors import SeamError
from seam.gitinfo import repo_root
from seam.jsonlog import log_event
from seam.powerstate import (
    assert_profile,
    capture_power_state,
    manifest_power_state,
    raise_if_profile_mismatch,
)
from seam.rawstore import RunDir

__all__ = [
    "BatterySample",
    "S1CharacterizationResult",
    "analyze_capacity_series",
    "poll_battery_status",
    "run_characterization",
]

_MEASUREMENT_CLASS: Final = "battery_counter_char"
_SAMPLES_NAME: Final = "samples.ndjson"


@dataclass(frozen=True, slots=True)
class BatterySample:
    """One WMI ``BatteryStatus`` poll."""

    t_ns: int
    remaining_capacity_mwh: float | None
    discharge_rate_mw: float | None
    voltage_mv: float | None
    sampler_overhead_us: int


@dataclass(frozen=True, slots=True)
class S1CharacterizationResult:
    """Derived M2.1 quantities. Distributions, not point estimates alone."""

    n_samples: int
    duration_s: float
    n_capacity_changes: int
    inter_change_intervals_s: list[float]
    capacity_deltas_mwh: list[float]
    update_period_median_s: float | None
    update_period_iqr_s: tuple[float, float] | None
    quantization_step_mwh_median: float | None
    quantization_step_mwh_iqr: tuple[float, float] | None
    min_resolvable_energy_mwh: float | None
    min_viable_energy_run_duration_s: float | None
    proposed_settle_s: float | None
    proposed_soc_window_pct: list[float] | None
    discharge_vs_delta_note: str


class _WmiBatterySession:
    """Persistent ``root\\WMI`` ``BatteryStatus`` session (amortizes COM init)."""

    def __init__(self) -> None:
        import pythoncom  # type: ignore[import-untyped]
        import wmi  # type: ignore[import-untyped]

        self._pythoncom = pythoncom
        self._pythoncom.CoInitialize()
        self._client = wmi.WMI(namespace="root\\WMI")

    def poll(self) -> tuple[float | None, float | None, float | None]:
        rows = list(self._client.BatteryStatus())
        if not rows:
            log_event(
                "s1.battery_status_empty",
                severity="warning",
                message="root\\WMI BatteryStatus returned no instances",
            )
            return None, None, None
        row = rows[0]
        return (
            _as_float(getattr(row, "RemainingCapacity", None)),
            _as_float(getattr(row, "DischargeRate", None)),
            _as_float(getattr(row, "Voltage", None)),
        )

    def close(self) -> None:
        self._pythoncom.CoUninitialize()


def poll_battery_status() -> tuple[float | None, float | None, float | None]:
    """Read ``RemainingCapacity``, ``DischargeRate``, ``Voltage`` from WMI.

    Returns:
        ``(remaining_mwh, discharge_mw, voltage_mv)``, any of which may be None.

    Raises:
        SeamError: If WMI is unavailable and this is not a mocked test path.
    """
    if sys.platform != "win32":
        raise SeamError("S1 battery characterization requires native Windows WMI")

    try:
        session = _WmiBatterySession()
    except ImportError:
        return _poll_via_powershell()
    try:
        return session.poll()
    finally:
        session.close()


def _poll_via_powershell() -> tuple[float | None, float | None, float | None]:
    """CIM fallback used when the ``wmi`` package is not installed."""
    import subprocess

    script = (
        "$b = Get-CimInstance -Namespace root/WMI -ClassName BatteryStatus -ErrorAction Stop | "
        "Select-Object -First 1; "
        "if (-not $b) { 'null,null,null' } else { "
        '"$($b.RemainingCapacity),$($b.DischargeRate),$($b.Voltage)" }'
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SeamError(f"PowerShell BatteryStatus query failed: {exc}") from exc
    if completed.returncode != 0:
        raise SeamError(
            f"PowerShell BatteryStatus query exited {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    parts = completed.stdout.strip().split(",")
    if len(parts) != 3:
        return None, None, None
    return _as_float(parts[0]), _as_float(parts[1]), _as_float(parts[2])


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "null", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def sample_once(
    *,
    clock_ns: Callable[[], int] | None = None,
    poll: Callable[[], tuple[float | None, float | None, float | None]] | None = None,
) -> BatterySample:
    """Take one timestamped sample; ``sampler_overhead_us`` is the poll duration."""
    clock = clock_ns or time.perf_counter_ns
    reader = poll or poll_battery_status
    t0 = clock()
    remaining, rate, voltage = reader()
    t1 = clock()
    return BatterySample(
        t_ns=t0,
        remaining_capacity_mwh=remaining,
        discharge_rate_mw=rate,
        voltage_mv=voltage,
        sampler_overhead_us=max(0, (t1 - t0) // 1_000),
    )


def iter_samples(
    *,
    duration_s: float,
    rate_hz: float,
    clock_ns: Callable[[], int] | None = None,
    poll: Callable[[], tuple[float | None, float | None, float | None]] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Iterator[BatterySample]:
    """Yield samples for ``duration_s`` at ``rate_hz``.

    Sleep happens **outside** the timed sample region (spec §3.5).
    When ``poll`` is omitted on Windows, a persistent WMI session is used so COM
    init is not paid on every sample (otherwise the poll alone exceeds a 10 Hz period).
    """
    if rate_hz <= 0:
        raise SeamError(f"sample_rate_hz must be positive; got {rate_hz}")
    if duration_s <= 0:
        raise SeamError(f"duration_s must be positive; got {duration_s}")

    clock = clock_ns or time.perf_counter_ns
    sleeper = sleep or time.sleep
    period_s = 1.0 / rate_hz
    t_end = clock() + int(duration_s * 1_000_000_000)

    session: _WmiBatterySession | None = None
    reader = poll
    if reader is None and sys.platform == "win32":
        try:
            session = _WmiBatterySession()
            reader = session.poll
        except ImportError:
            reader = poll_battery_status

    try:
        while clock() < t_end:
            sample = sample_once(clock_ns=clock, poll=reader)
            yield sample
            # Sleep for the remainder of the period; never sleep inside the timed poll.
            elapsed_s = (clock() - sample.t_ns) / 1_000_000_000
            remaining = period_s - elapsed_s
            if remaining > 0:
                sleeper(remaining)
    finally:
        if session is not None:
            session.close()


def _sample_to_record(sample: BatterySample) -> dict[str, Any]:
    return {
        "t_ns": sample.t_ns,
        "batt_remaining_mwh": sample.remaining_capacity_mwh,
        "batt_discharge_mw": sample.discharge_rate_mw,
        "batt_voltage_mv": sample.voltage_mv,
        "sampler_overhead_us": sample.sampler_overhead_us,
    }


def write_samples_ndjson(run_dir: RunDir, samples: Sequence[BatterySample]) -> Path:
    """Append samples to ``samples.ndjson`` inside an open (unsealed) run directory."""
    path = run_dir.path / _SAMPLES_NAME
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for sample in samples:
            fh.write(json.dumps(_sample_to_record(sample), separators=(",", ":")) + "\n")
    return path


def load_samples_ndjson(path: Path) -> list[BatterySample]:
    """Load ``samples.ndjson`` written by :func:`write_samples_ndjson`."""
    samples: list[BatterySample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(
            BatterySample(
                t_ns=int(row["t_ns"]),
                remaining_capacity_mwh=row.get("batt_remaining_mwh"),
                discharge_rate_mw=row.get("batt_discharge_mw"),
                voltage_mv=row.get("batt_voltage_mv"),
                sampler_overhead_us=int(row["sampler_overhead_us"]),
            )
        )
    return samples


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        raise SeamError("percentile of empty series")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def analyze_capacity_series(samples: Sequence[BatterySample]) -> S1CharacterizationResult:
    """Derive M2.1 quantities from a capacity time series (pure; hardware not required)."""
    if len(samples) < 2:
        raise SeamError("need at least two samples to characterize the battery counter")

    duration_s = (samples[-1].t_ns - samples[0].t_ns) / 1_000_000_000
    intervals: list[float] = []
    deltas: list[float] = []
    prev_cap: float | None = None
    prev_t: int | None = None
    for sample in samples:
        cap = sample.remaining_capacity_mwh
        if cap is None:
            continue
        if prev_cap is not None and prev_t is not None and cap != prev_cap:
            intervals.append((sample.t_ns - prev_t) / 1_000_000_000)
            deltas.append(abs(cap - prev_cap))
            prev_t = sample.t_ns
            prev_cap = cap
        elif prev_cap is None:
            prev_cap = cap
            prev_t = sample.t_ns

    def summary(values: list[float]) -> tuple[float | None, tuple[float, float] | None]:
        if not values:
            return None, None
        ordered = sorted(values)
        med = statistics.median(ordered)
        return med, (_percentile(ordered, 0.25), _percentile(ordered, 0.75))

    period_med, period_iqr = summary(intervals)
    step_med, step_iqr = summary(deltas)

    min_resolvable = min(deltas) if deltas else step_med
    min_viable: float | None = None
    if period_med is not None and period_med > 0:
        # Edge-effect bound: uncertainty ~ period/duration; <=5% => duration >= 20*period.
        # Not "20 energy quanta" - the EC is time-driven (see AUDIT_LOG M2.1 closeout Item 1).
        min_viable = 20.0 * period_med

    # Settling heuristic for the *proposal* before the real unplug measurement: none yet.
    proposed_settle = None
    proposed_soc = None

    note = (
        "S1 energy estimator is ΔRemainingCapacity (AM-018, PRE-DATA w.r.t. M2.5). "
        "Integrated DischargeRate is a cross-check only; per-update mWh increments are "
        "power-dependent, not a fixed device quantum."
    )

    return S1CharacterizationResult(
        n_samples=len(samples),
        duration_s=duration_s,
        n_capacity_changes=len(deltas),
        inter_change_intervals_s=intervals,
        capacity_deltas_mwh=deltas,
        update_period_median_s=period_med,
        update_period_iqr_s=period_iqr,
        quantization_step_mwh_median=step_med,
        quantization_step_mwh_iqr=step_iqr,
        min_resolvable_energy_mwh=min_resolvable,
        min_viable_energy_run_duration_s=min_viable,
        proposed_settle_s=proposed_settle,
        proposed_soc_window_pct=proposed_soc,
        discharge_vs_delta_note=note,
    )


def _logical_cpus_from_config(config: ResolvedConfig) -> list[int]:
    """Return verified logical CPU indices - never ``os.cpu_count()`` (seam.mdc)."""
    topo = config.get("topology") or {}
    if not topo.get("verified"):
        raise SeamError("synthetic load requires a verified topology mapping (p_cpus + lpe_cpus)")
    p_cpus = list(topo.get("p_cpus") or [])
    lpe_cpus = list(topo.get("lpe_cpus") or [])
    cpus = [int(c) for c in (*p_cpus, *lpe_cpus)]
    if not cpus:
        raise SeamError("verified topology has empty p_cpus/lpe_cpus")
    return cpus


def _spin_worker(cpu: int, stop_event: Any) -> None:
    """Busy-wait on one logical CPU until ``stop_event`` is set."""
    try:
        import psutil

        psutil.Process().cpu_affinity([cpu])
    except Exception:
        # Affinity is best-effort; the busy loop still burns cycles unpinned.
        pass
    n = 0
    while not stop_event.is_set():
        n = (n + 1) & 0xFFFFFFFF
        if n == 0 and stop_event.is_set():
            break


def start_synthetic_load(cpus: Sequence[int]) -> tuple[Any, list[mp.Process]]:
    """Start one busy-loop process per logical CPU. Returns (stop_event, processes)."""
    stop_event = mp.Event()
    procs: list[mp.Process] = []
    for cpu in cpus:
        proc = mp.Process(
            target=_spin_worker,
            args=(int(cpu), stop_event),
            name=f"s1-spin-cpu{cpu}",
            daemon=True,
        )
        proc.start()
        procs.append(proc)
    log_event(
        "s1.synthetic_load_started",
        message=f"started {len(procs)} spin workers on logical CPUs {list(cpus)}",
        cpus=list(cpus),
    )
    return stop_event, procs


def stop_synthetic_load(stop_event: Any, procs: Sequence[mp.Process]) -> None:
    stop_event.set()
    for proc in procs:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
    log_event("s1.synthetic_load_stopped", message=f"stopped {len(procs)} spin workers")


def run_characterization(
    *,
    config: ResolvedConfig,
    duration_s: float,
    allow_dirty: bool,
    repo: Path,
    background_quiesced: bool = True,
    poll: Callable[[], tuple[float | None, float | None, float | None]] | None = None,
    with_synthetic_load: bool = True,
) -> dict[str, Any]:
    """Assert battery-pinned, sample under load, and seal a run.

    When the profile mismatches, emits a refusal manifest and raises
    :class:`ProfileMismatchError`. Samples are written **before** the integrity hash
    (via ``emit(..., before_integrity_hash=...)``) so they are covered by ``raw_sha256``.
    """
    from seam.manifest import emit

    power_cfg = config.get("power") or {}
    s1_cfg = config.get("s1_characterization") or {}
    rate_hz = float(s1_cfg.get("sample_rate_hz") or 10)
    load_name = str(s1_cfg.get("synthetic_load") or "cpu_spin_all_logical")

    power_state = capture_power_state()
    assertion = assert_profile(
        _MEASUREMENT_CLASS,
        power_state,
        power_cfg=power_cfg,
        background_quiesced=background_quiesced,
    )
    matched = not assertion.deviations

    # Mutable: end-of-run SoC filled after sampling, before build_manifest.
    power_block = manifest_power_state(
        power_state,
        battery_pct_end=None,
        profile=assertion,
        background_quiesced=background_quiesced,
    )

    base_summary: dict[str, Any] = {
        "measurement_class": _MEASUREMENT_CLASS,
        "profile": asdict(assertion),
        "sample_rate_hz": rate_hz,
        "requested_duration_s": duration_s,
        "synthetic_load": load_name if (matched and with_synthetic_load) else None,
        "verdict": "pass" if matched else "refuse_profile_mismatch",
        "soc_at_start_note": (
            f"SoC at start={power_state.battery_pct}% "
            "(proposed window [40,85] not yet pinned; recorded as covariate)"
        ),
    }

    def _collect(run_dir: RunDir) -> dict[str, Any] | None:
        if not matched:
            return None
        stop_event = None
        procs: list[mp.Process] = []
        try:
            if with_synthetic_load:
                if load_name != "cpu_spin_all_logical":
                    raise SeamError(
                        f"unsupported synthetic_load {load_name!r}; "
                        f"only 'cpu_spin_all_logical' is implemented for M2.1"
                    )
                cpus = _logical_cpus_from_config(config)
                stop_event, procs = start_synthetic_load(cpus)
                time.sleep(5.0)
            # Stream samples to disk as they arrive so a mid-run crash does not lose the series.
            samples: list[BatterySample] = []
            path = run_dir.path / _SAMPLES_NAME
            with path.open("a", encoding="utf-8", newline="\n", buffering=1) as fh:
                for sample in iter_samples(duration_s=duration_s, rate_hz=rate_hz, poll=poll):
                    samples.append(sample)
                    fh.write(json.dumps(_sample_to_record(sample), separators=(",", ":")) + "\n")
                    if len(samples) % 100 == 0:
                        fh.flush()
                        log_event(
                            "s1.sample_progress",
                            message=f"wrote {len(samples)} samples",
                            n_samples=len(samples),
                        )
        finally:
            if stop_event is not None:
                stop_synthetic_load(stop_event, procs)

        end_power = capture_power_state()
        power_block["battery_pct_end"] = end_power.battery_pct
        power_block["soc_at_end"] = end_power.battery_pct

        analysis = analyze_capacity_series(samples)
        return {
            "n_samples": analysis.n_samples,
            "duration_s": analysis.duration_s,
            "n_capacity_changes": analysis.n_capacity_changes,
            "update_period_median_s": analysis.update_period_median_s,
            "update_period_iqr_s": analysis.update_period_iqr_s,
            "quantization_step_mwh_median": analysis.quantization_step_mwh_median,
            "quantization_step_mwh_iqr": analysis.quantization_step_mwh_iqr,
            "min_resolvable_energy_mwh": analysis.min_resolvable_energy_mwh,
            "min_viable_energy_run_duration_s": analysis.min_viable_energy_run_duration_s,
            "discharge_vs_delta_note": analysis.discharge_vs_delta_note,
            "inter_change_intervals_s": analysis.inter_change_intervals_s,
            "capacity_deltas_mwh": analysis.capacity_deltas_mwh,
            "soc_at_end": end_power.battery_pct,
            "on_battery_end": end_power.on_battery,
            "charging_end": end_power.charging,
        }

    run = emit(
        config=config,
        target="cpu-p",
        workload={
            "kind": "battery_counter_char",
            "benchmark": "s1_wmi_battery_status_v1",
            "task_ids": [],
            "seed": None,
            "n_repeats": None,
        },
        condition_label="battery_counter_char",
        allow_dirty=allow_dirty,
        repo_root=repo,
        summary=base_summary,
        power_state=power_block,
        outputs={"samples": _SAMPLES_NAME} if matched else None,
        self_check="pass" if matched else "fail",
        before_integrity_hash=_collect if matched else None,
    )

    if not matched:
        raise_if_profile_mismatch(assertion)

    return {
        "run_id": run.run_id,
        "verdict": "pass",
        "summary": json.loads((run.run_dir.path / "summary.json").read_text(encoding="utf-8")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-id", default="aipc-c1")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="override s1_characterization.proposed_duration_s",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--assert-only",
        action="store_true",
        help="assert battery-pinned and emit refuse/pass manifest without sampling",
    )
    parser.add_argument(
        "--background-quiesced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="operator attestation that background clients are quiesced",
    )
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    config = load_platform_config(args.platform_id, repo_root=root)
    duration = args.duration_s
    if duration is None:
        s1 = config.get("s1_characterization") or {}
        duration = float(s1.get("proposed_duration_s") or 1800)

    if args.assert_only:
        # Emit refusal or success without sampling - for suite / dry checks on AC.
        from seam.manifest import emit

        power_state = capture_power_state()
        assertion = assert_profile(
            _MEASUREMENT_CLASS,
            power_state,
            power_cfg=config.get("power") or {},
            background_quiesced=args.background_quiesced,
        )
        end = capture_power_state()
        matched = not assertion.deviations
        run = emit(
            config=config,
            target="cpu-p",
            workload={
                "kind": "battery_counter_char",
                "benchmark": "s1_wmi_battery_status_v1",
                "task_ids": [],
                "seed": None,
                "n_repeats": None,
            },
            condition_label="battery_counter_char_assert_only",
            allow_dirty=args.allow_dirty,
            repo_root=root,
            summary={
                "measurement_class": _MEASUREMENT_CLASS,
                "profile": asdict(assertion),
                "verdict": "pass" if matched else "refuse_profile_mismatch",
                "assert_only": True,
            },
            power_state=manifest_power_state(
                power_state,
                battery_pct_end=end.battery_pct,
                profile=assertion,
                background_quiesced=args.background_quiesced,
            ),
            self_check="pass" if matched else "fail",
        )
        print(f"run_id={run.run_id} matched={matched} deviations={assertion.deviations}")
        if not matched:
            raise_if_profile_mismatch(assertion)
        return 0

    # Full measurement path - must not run until the human confirms unplugged.
    result = run_characterization(
        config=config,
        duration_s=duration,
        allow_dirty=args.allow_dirty,
        repo=root,
        background_quiesced=args.background_quiesced,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
