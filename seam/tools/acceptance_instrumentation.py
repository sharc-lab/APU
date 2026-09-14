"""Additive instruments for the fixed-throughput acceptance gate.

These sit alongside the throughput endpoint; they do not replace it and they do not change the
derived band. Recorded so a near-1.0 or far-from-1.0 ratio can be read against memory dispersion,
sleep/power transitions across the gap, network reachability, and whether PCORE_ONLY actually
confined the work on the cores the config asked for.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from typing import Any

from seam.tools.confinement_classify import classify_cell, compute_noise_band

__all__ = [
    "capture_arm_memory",
    "capture_uptime_wake",
    "classify_arm_confinement",
    "collect_kernel_power_events",
    "correlate_kernel_power_with_repeats",
    "host_memory_snapshot",
    "memory_dispersion",
    "parse_utc",
    "reachability_sampler",
    "summarize_power_gap",
]


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def host_memory_snapshot() -> dict[str, Any]:
    """Free / available physical memory and commit charge of the host at one instant."""
    import psutil

    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "observed_utc": _utc(),
        "free_physical_bytes": int(virtual.free),
        "available_bytes": int(virtual.available),
        "total_physical_bytes": int(virtual.total),
        "commit_total_bytes": int(getattr(swap, "total", 0) or 0),
        "commit_used_bytes": int(getattr(swap, "used", 0) or 0),
    }


def capture_arm_memory(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Peak working set / peak commit / free physical at start and at the peak sample.

    Working set and commit come from the child's RssSampler window (peak_rss_bytes /
    peak_commit_bytes). Free physical at start is the host snapshot before the first repeat;
    free physical at peak is the free_physical_mb_at_peak sample taken when the process hit its
    RSS high-water, converted back to bytes.
    """
    peaks_ws: list[int] = []
    peaks_commit: list[int] = []
    free_at_peak_mb: list[float] = []
    free_at_start_mb: list[float] = []

    for record in records:
        generation = (record.get("result", {}).get("child") or {}).get("generation") or {}
        ws = generation.get("peak_rss_bytes")
        commit = generation.get("peak_commit_bytes")
        if ws is not None:
            peaks_ws.append(int(ws))
        if commit is not None:
            peaks_commit.append(int(commit))
        peak_free = generation.get("free_physical_mb_at_peak")
        if peak_free is not None:
            free_at_peak_mb.append(float(peak_free))
        start_free = generation.get("free_physical_mb_start")
        if start_free is None:
            start_free = generation.get("free_memory_mb_start")
        if start_free is not None:
            free_at_start_mb.append(float(start_free))

    return {
        "peak_working_set_bytes": max(peaks_ws) if peaks_ws else None,
        "peak_commit_bytes": max(peaks_commit) if peaks_commit else None,
        "free_physical_bytes_at_start": (
            int(min(free_at_start_mb) * 1024 * 1024) if free_at_start_mb else None
        ),
        "free_physical_bytes_at_peak": (
            int(min(free_at_peak_mb) * 1024 * 1024) if free_at_peak_mb else None
        ),
        "per_repeat_peak_working_set_bytes": peaks_ws,
        "per_repeat_peak_commit_bytes": peaks_commit,
    }


def memory_dispersion(arm_a: dict[str, Any], arm_b: dict[str, Any]) -> dict[str, Any]:
    """Ratio and absolute delta of the additive memory metrics across the two arms."""

    def _ratio(a: int | None, b: int | None) -> float | None:
        if a is None or b is None or min(a, b) <= 0:
            return None
        return max(a, b) / min(a, b)

    def _delta(a: int | None, b: int | None) -> int | None:
        if a is None or b is None:
            return None
        return int(b) - int(a)

    keys = (
        "peak_working_set_bytes",
        "peak_commit_bytes",
        "free_physical_bytes_at_start",
        "free_physical_bytes_at_peak",
    )
    return {
        key: {
            "arm_1": arm_a.get(key),
            "arm_2": arm_b.get(key),
            "ratio_max_over_min": _ratio(arm_a.get(key), arm_b.get(key)),
            "signed_delta_arm2_minus_arm1": _delta(arm_a.get(key), arm_b.get(key)),
        }
        for key in keys
    }


def capture_uptime_wake() -> dict[str, Any]:
    """System uptime and last-wake timestamp. Never changes the power policy."""
    import psutil

    boot_s = float(psutil.boot_time())
    now_s = time.time()
    record: dict[str, Any] = {
        "observed_utc": _utc(),
        "boot_time_utc": datetime.fromtimestamp(boot_s, UTC).isoformat(),
        "uptime_s": now_s - boot_s,
        "last_wake_utc": None,
        "last_wake_source": None,
        "last_wake_error": None,
    }
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "$e = Get-WinEvent -FilterHashtable @{"
                    "LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=507}"
                    " -MaxEvents 1 -ErrorAction SilentlyContinue; "
                    "if ($e) { $e.TimeCreated.ToUniversalTime().ToString('o') }"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = (completed.stdout or "").strip()
        if completed.returncode == 0 and text:
            record["last_wake_utc"] = text
            record["last_wake_source"] = "Kernel-Power/507"
        else:
            record["last_wake_error"] = (
                f"returncode={completed.returncode}; stderr={(completed.stderr or '')[-400:]}"
            )
    except Exception as exc:
        record["last_wake_error"] = f"{type(exc).__name__}: {exc}"
    return record


def parse_utc(value: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp; accept trailing ``Z``."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def collect_kernel_power_events(*, start_utc: str, end_utc: str) -> dict[str, Any]:
    """Kernel-Power entries whose TimeCreated falls in ``[start_utc, end_utc]``.

    Sleep/resume transitions are the question. The power policy is never modified.
    Each event keeps timestamp, Id, and payload (message + property values) - not only a count.
    """
    # Event ids commonly associated with sleep/resume on modern Windows.
    transition_ids = (42, 107, 506, 507, 566)
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    f"$start = [datetimeoffset]::Parse('{start_utc}').UtcDateTime; "
                    f"$end = [datetimeoffset]::Parse('{end_utc}').UtcDateTime; "
                    "$events = Get-WinEvent -FilterHashtable @{"
                    "LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'"
                    "} -ErrorAction SilentlyContinue | "
                    "Where-Object { $_.TimeCreated.ToUniversalTime() -ge $start "
                    "-and $_.TimeCreated.ToUniversalTime() -le $end }; "
                    "if (-not $events) { '[]'; exit 0 }; "
                    "$events | Select-Object "
                    "@{n='time_utc';e={$_.TimeCreated.ToUniversalTime().ToString('o')}}, "
                    "Id, LevelDisplayName, Message, "
                    "@{n='properties';e={@($_.Properties | ForEach-Object { $_.Value })}} "
                    "| ConvertTo-Json -Depth 6 -Compress"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        raw = (completed.stdout or "").strip()
        if completed.returncode != 0:
            return {
                "start_utc": start_utc,
                "end_utc": end_utc,
                "events": [],
                "transition_occurred": None,
                "error": f"returncode={completed.returncode}; stderr={(completed.stderr or '')[-600:]}",
            }
        import json

        parsed = json.loads(raw) if raw else []
        if isinstance(parsed, dict):
            parsed = [parsed]
        events = []
        for row in parsed:
            message = row.get("Message") or ""
            props = row.get("properties")
            if props is not None and not isinstance(props, list):
                props = [props]
            events.append(
                {
                    "time_utc": row.get("time_utc"),
                    "id": row.get("Id"),
                    "level": row.get("LevelDisplayName"),
                    "message_head": message[:240],
                    "payload": {
                        "message": message,
                        "properties": props,
                    },
                }
            )
        transition_occurred = any(
            int(e["id"]) in transition_ids for e in events if e.get("id") is not None
        )
        return {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "events": events,
            "transition_event_ids_watched": list(transition_ids),
            "transition_occurred": transition_occurred,
            "error": None,
        }
    except Exception as exc:
        return {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "events": [],
            "transition_occurred": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def correlate_kernel_power_with_repeats(
    *,
    events: list[dict[str, Any]],
    repeats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag each repeat window for any Kernel-Power event whose timestamp falls inside it.

    ``repeats`` entries need ``start_utc`` / ``end_utc`` (inclusive). This is the decisive
    confound test: outlier repeats that contain transition events while clean repeats do not.
    """
    parsed_events: list[tuple[datetime, dict[str, Any]]] = []
    for event in events:
        ts = parse_utc(event.get("time_utc"))
        if ts is not None:
            parsed_events.append((ts, event))

    rows: list[dict[str, Any]] = []
    for repeat in repeats:
        start = parse_utc(repeat.get("start_utc"))
        end = parse_utc(repeat.get("end_utc"))
        inside: list[dict[str, Any]] = []
        if start is not None and end is not None:
            for ts, event in parsed_events:
                if start <= ts <= end:
                    inside.append(
                        {
                            "time_utc": event.get("time_utc"),
                            "id": event.get("id"),
                            "message_head": event.get("message_head"),
                        }
                    )
        rows.append(
            {
                **repeat,
                "kernel_power_event_in_window": bool(inside),
                "kernel_power_events_in_window": inside,
                "kernel_power_event_count_in_window": len(inside),
            }
        )
    return rows


def summarize_power_gap(
    *,
    arm1_uptime: dict[str, Any],
    arm2_uptime: dict[str, Any],
    kernel_power: dict[str, Any],
) -> dict[str, Any]:
    """Whether any sleep/resume transition occurred across the acceptance window."""
    boot_changed = arm1_uptime.get("boot_time_utc") != arm2_uptime.get("boot_time_utc")
    wake_changed = (
        arm1_uptime.get("last_wake_utc") is not None
        and arm2_uptime.get("last_wake_utc") is not None
        and arm1_uptime.get("last_wake_utc") != arm2_uptime.get("last_wake_utc")
    )
    events = list(kernel_power.get("events") or [])
    transition = bool(kernel_power.get("transition_occurred")) or boot_changed or wake_changed
    return {
        "transition_occurred": transition,
        "boot_time_changed": boot_changed,
        "last_wake_changed": wake_changed,
        "kernel_power_transition_occurred": kernel_power.get("transition_occurred"),
        "kernel_power_event_count": len(events),
        # Full list - count alone cannot locate which repeat absorbed a transition.
        "kernel_power_events": events,
        "arm1_uptime": arm1_uptime,
        "arm2_uptime": arm2_uptime,
        "note": (
            "Power policy was not modified. A transition here is a confound for the throughput "
            "ratio, not a criterion the harness adjusts away."
        ),
    }


class _ReachabilitySampler:
    def __init__(self, *, host: str, interval_s: float, log_path: Any) -> None:
        self._host = host
        self._interval_s = interval_s
        self._log_path = log_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="acceptance-reach", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 5.0)
            self._thread = None
        payload = {
            "host": self._host,
            "interval_s": self._interval_s,
            "samples": list(self._samples),
            "n_samples": len(self._samples),
            "n_unreachable": sum(1 for s in self._samples if not s.get("reachable")),
        }
        try:
            import json
            from pathlib import Path

            path = Path(self._log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            payload["log_path"] = str(path)
        except Exception as exc:
            payload["log_path_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._probe_once()
            self._samples.append(sample)
            self._stop.wait(self._interval_s)

    def _probe_once(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                ["ping", "-n", "1", "-w", "1000", self._host],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            reachable = completed.returncode == 0
            return {
                "observed_utc": _utc(),
                "reachable": reachable,
                "returncode": completed.returncode,
                "rtt_s": time.perf_counter() - started if reachable else None,
            }
        except Exception as exc:
            return {
                "observed_utc": _utc(),
                "reachable": False,
                "returncode": None,
                "rtt_s": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


def reachability_sampler(*, host: str, interval_s: float, log_path: Any) -> _ReachabilitySampler:
    return _ReachabilitySampler(host=host, interval_s=interval_s, log_path=log_path)


def classify_arm_confinement(
    *,
    p_cpus: list[int],
    lpe_cpus: list[int],
    util_series: list[list[float]],
    baseline_series: list[list[float]],
) -> dict[str, Any]:
    """Classify whether work stayed on ``p_cpus`` under PCORE_ONLY, using confinement_classify.

    Records the verdict that actually happened. Does not change the config to manufacture
    CONFINED. Thresholds use the legacy single-threshold path (half the median delta on the
    requested set) because this gate has no A5/A6 reference cells - those belong to the
    affinity matrix, not the acceptance arm.
    """
    all_cpus = {int(c) for c in (*p_cpus, *lpe_cpus)}
    requested = {int(c) for c in p_cpus}

    if not util_series or not baseline_series:
        return {
            "verdict": "INVALID",
            "reason": "insufficient utilization samples for confinement classification",
            "requested_cpus": sorted(requested),
            "n_util_samples": len(util_series),
            "n_baseline_samples": len(baseline_series),
        }

    n_cores = max(len(row) for row in util_series)
    baseline_means_per_core: dict[int, list[float]] = {c: [] for c in range(n_cores)}
    for row in baseline_series:
        for idx, value in enumerate(row):
            baseline_means_per_core[idx].append(float(value))
    # compute_noise_band expects per-core lists of means across scored generations; here each
    # sample is one observation, so treat each sample mean-equivalent as one entry.
    noise_band = compute_noise_band(baseline_means_per_core)

    baseline_mean = [
        statistics.fmean(baseline_means_per_core[c]) if baseline_means_per_core[c] else 0.0
        for c in range(n_cores)
    ]
    util_mean = [
        statistics.fmean(float(row[c]) for row in util_series if c < len(row))
        for c in range(n_cores)
    ]
    deltas = {c: util_mean[c] - baseline_mean[c] for c in range(n_cores)}

    cell = classify_cell(
        deltas,
        requested,
        noise_band=noise_band,
        all_cpus=all_cpus,
    )
    return {
        "verdict": cell["verdict"],
        "requested_cpus": cell["requested_cpus"],
        "noise_band": cell["noise_band"],
        "loaded_threshold": cell["loaded_threshold"],
        "core_states": cell["core_states"],
        "deltas": {str(k): v for k, v in sorted(deltas.items())},
        "baseline_mean": baseline_mean,
        "util_mean": util_mean,
        "n_util_samples": len(util_series),
        "n_baseline_samples": len(baseline_series),
        "config_claim": "SCHEDULING_CORE_TYPE=PCORE_ONLY (control, not an adoption claim)",
        "note": (
            "Verdict records what happened under the configured properties. UNCLEAR/LEAKED/"
            "INVALID here do not rewrite the config; they travel with the sealed arm."
        ),
    }
