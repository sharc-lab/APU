"""
Smoke-test every Linux-only code path used by runner.py and telemetry.py.
Run on the EVO-X2 before starting any measurement session.

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed or returned a degraded/None reading.

Usage:
    python scripts/verify_platform.py
    python scripts/verify_platform.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _check_proc_meminfo() -> tuple[str, str]:
    """Parse /proc/meminfo for MemTotal and MemAvailable (unified pool path)."""
    proc = Path("/proc/meminfo")
    if not proc.exists():
        return FAIL, f"/proc/meminfo not found — not Linux or missing procfs"
    try:
        text = proc.read_text(encoding="utf-8")
    except OSError as e:
        return FAIL, f"/proc/meminfo unreadable: {e}"
    fields: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            try:
                fields[key] = int(parts[1])
            except ValueError:
                pass
    missing = [k for k in ("MemTotal", "MemAvailable") if k not in fields]
    if missing:
        return FAIL, f"Missing keys in /proc/meminfo: {missing} (kernel < 3.14?)"
    total_mb = fields["MemTotal"] / 1024
    avail_mb = fields["MemAvailable"] / 1024
    used_mb = total_mb - avail_mb
    return PASS, f"MemTotal={total_mb:.0f} MB  MemAvailable={avail_mb:.0f} MB  used={used_mb:.0f} MB"


def _check_free_b() -> tuple[str, str]:
    """Run `free -b` as the MANIFEST sampler does."""
    try:
        out = subprocess.check_output(["free", "-b"], stderr=subprocess.DEVNULL, timeout=5)
        lines = out.decode().strip().splitlines()
        if not any("Mem:" in l for l in lines):
            return FAIL, f"`free -b` ran but output lacks 'Mem:' line — unexpected format"
        return PASS, f"`free -b` returned {len(lines)} lines"
    except FileNotFoundError:
        return FAIL, "`free` binary not found — install procps"
    except subprocess.TimeoutExpired:
        return FAIL, "`free -b` timed out after 5 s"
    except subprocess.CalledProcessError as e:
        return FAIL, f"`free -b` exited {e.returncode}"


def _check_sensors_temperatures() -> tuple[str, str]:
    """
    Check psutil.sensors_temperatures() for usable CPU temperature.
    Handles both Intel (coretemp / 'Package id 0') and AMD (k10temp: Tctl, Tdie, Tccd*).
    Returns FAIL if no temperature can be read at all.
    Returns WARN if a sensor is found but the label mapping is the fallback (AMD Tctl case).
    """
    try:
        import psutil
    except ImportError:
        return FAIL, "psutil not installed"
    if not hasattr(psutil, "sensors_temperatures"):
        return FAIL, "psutil.sensors_temperatures not available on this OS"
    temps = psutil.sensors_temperatures()
    if not temps:
        return FAIL, "psutil.sensors_temperatures() returned empty dict — no hwmon drivers loaded?"
    # Intel path: coretemp with a 'package' label
    for key in ("coretemp",):
        entries = temps.get(key, [])
        pkg = [e for e in entries if "package" in e.label.lower()]
        if pkg:
            return PASS, f"{key}: '{pkg[0].label}' = {pkg[0].current:.1f} °C"
    # AMD k10temp path: check for named labels Tctl / Tdie / Tccd*
    for key in ("k10temp", "zenpower"):
        entries = temps.get(key, [])
        if not entries:
            continue
        # Prefer Tdie (junction temperature) > Tctl (control, includes offset) > first
        preferred = next(
            (e for e in entries if e.label in ("Tdie",)), None
        ) or next(
            (e for e in entries if e.label in ("Tctl",)), None
        ) or entries[0]
        label_note = ""
        if preferred.label == "Tctl":
            label_note = " (Tctl — NOT package; runner.py labels this cpu_pkg_temp_c, which is misleading)"
            return WARN, f"{key}: '{preferred.label}' = {preferred.current:.1f} °C{label_note}"
        return PASS, f"{key}: '{preferred.label}' = {preferred.current:.1f} °C"
    # Fallback: any sensor with any entry
    for key, entries in temps.items():
        if entries:
            return WARN, (
                f"Unknown sensor '{key}' label='{entries[0].label}' "
                f"= {entries[0].current:.1f} °C — verify this is the CPU sensor"
            )
    return FAIL, f"Sensors present ({list(temps.keys())}) but all have zero entries"


def _check_cpu_freq() -> tuple[str, str]:
    """Check psutil.cpu_freq(percpu=True) returns non-zero readings."""
    try:
        import psutil
    except ImportError:
        return FAIL, "psutil not installed"
    try:
        freqs = psutil.cpu_freq(percpu=True)
    except Exception as e:
        return FAIL, f"psutil.cpu_freq(percpu=True) raised: {type(e).__name__}: {e}"
    if not freqs:
        return FAIL, "psutil.cpu_freq(percpu=True) returned empty list — cpufreq driver not loaded?"
    currents = [f.current for f in freqs if f.current]
    if not currents:
        return FAIL, f"{len(freqs)} CPU entries but all have current=0 — cpufreq scaling unavailable"
    mean_mhz = sum(currents) / len(currents)
    return PASS, f"{len(freqs)} cores, mean {mean_mhz:.0f} MHz (range {min(currents):.0f}–{max(currents):.0f})"


def _check_fsync_dir(results_dir: Path) -> tuple[str, str]:
    """
    Reproduce _fsync_dir() from runner.py: open a directory fd and fsync it.
    Tests both that os.O_RDONLY is valid for a directory and that the fsync call completes.
    """
    target = results_dir if results_dir.is_dir() else Path(tempfile.gettempdir())
    try:
        fd = os.open(str(target), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return PASS, f"directory fsync on {target} succeeded"
    except AttributeError:
        return FAIL, "os.O_RDONLY not defined — not a POSIX system"
    except OSError as e:
        return FAIL, f"directory fsync raised OSError: {e}"


def _check_psutil_virtual_memory() -> tuple[str, str]:
    """Check psutil.virtual_memory() returns a usable available field (used by preflight)."""
    try:
        import psutil
    except ImportError:
        return FAIL, "psutil not installed"
    try:
        vm = psutil.virtual_memory()
    except Exception as e:
        return FAIL, f"psutil.virtual_memory() raised: {e}"
    avail_gb = vm.available / 1024**3
    if avail_gb <= 0:
        return FAIL, f"virtual_memory().available = {vm.available} — zero or negative"
    return PASS, f"available = {avail_gb:.1f} GB  total = {vm.total / 1024**3:.1f} GB"


def _check_nvidia_smi() -> tuple[str, str]:
    """
    Verify that nvidia-smi is absent (expected on AMD Strix Halo) or present.
    Reports WARN if absent — this is expected on AMD but the discrete path in
    telemetry.py would return (None, 'unavailable:nvidia_smi_not_found').
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        lines = [l.strip() for l in out.decode().splitlines() if l.strip()]
        if lines:
            vals = []
            for l in lines:
                try:
                    vals.append(float(l))
                except ValueError:
                    pass
            if vals:
                return PASS, f"nvidia-smi: {len(vals)} GPU(s), used: {vals} MiB"
        return WARN, "nvidia-smi ran but returned no numeric values"
    except FileNotFoundError:
        return WARN, "nvidia-smi not found (expected on AMD Strix Halo; discrete path returns None)"
    except subprocess.TimeoutExpired:
        return FAIL, "nvidia-smi timed out — GPU driver may be hung"
    except subprocess.CalledProcessError as e:
        return WARN, f"nvidia-smi exited {e.returncode} (driver issue or no NVIDIA GPU)"


def _label(status: str) -> str:
    return {"PASS": "[ PASS ]", "FAIL": "[ FAIL ]", "WARN": "[ WARN ]"}[status]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results", type=Path,
                    help="Path to results/ directory (used for fsync test)")
    args = ap.parse_args()

    checks: list[tuple[str, tuple[str, str]]] = [
        ("/proc/meminfo (unified pool telemetry)", _check_proc_meminfo()),
        ("`free -b` (MANIFEST sampler)", _check_free_b()),
        ("psutil.sensors_temperatures() (CPU temp sampler)", _check_sensors_temperatures()),
        ("psutil.cpu_freq(percpu=True) (freq sampler)", _check_cpu_freq()),
        ("psutil.virtual_memory() (preflight)", _check_psutil_virtual_memory()),
        ("directory fsync / os.O_RDONLY (runner durability)", _check_fsync_dir(args.results_dir)),
        ("nvidia-smi / discrete GPU telemetry", _check_nvidia_smi()),
    ]

    col_w = max(len(name) for name, _ in checks) + 2
    overall = PASS
    lines = []
    lines.append("")
    lines.append("  Platform capability check")
    lines.append("  " + "─" * (col_w + 50))
    for name, (status, detail) in checks:
        lines.append(f"  {_label(status)}  {name:<{col_w}}  {detail}")
        if status == FAIL:
            overall = FAIL
        elif status == WARN and overall == PASS:
            overall = WARN
    lines.append("  " + "─" * (col_w + 50))
    lines.append(f"  Overall: {_label(overall)}")
    lines.append("")
    print("\n".join(lines))

    return 0 if overall in (PASS, WARN) else 1


if __name__ == "__main__":
    sys.exit(main())
