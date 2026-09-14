"""Per-call telemetry: wall time, TTFT, token counts, RSS, GPU/unified memory.

gpu_mem_mb field semantics by source
--------------------------------------
"nvidia-smi"        Discrete NVIDIA VRAM *currently in use* by all processes on
                    GPU 0.  Unit: MiB.  Meaningful only on dedicated VRAM; not
                    comparable to unified-memory figures.

"rocm-smi"          AMD GPU VRAM Total Used Memory in bytes, converted to MiB.
                    On AMD Strix Halo (unified LPDDR5X) this is the GPU driver's
                    view of its allocation within the shared pool — not
                    whole-system usage and not directly comparable to NVIDIA
                    discrete readings.

"intel-level-zero"  Intel GPU memory-module state via Level Zero Sysman API
                    (zesMemoryGetState, summed across all memory modules).  On
                    integrated Arc (unified pool) this is GPU-allocated shared
                    DRAM.  On discrete Arc this would be dedicated VRAM.  Not
                    directly comparable to NVIDIA discrete readings.

"intel-sysfs"       Intel i915 debug-fs or DRM sysfs memory counters on Linux.
                    Requires debugfs mount and read access; usually unavailable
                    without root.  Value is GPU GEM object total in MiB.

"unified-psutil"    Whole-system used memory = total − available (psutil).
                    This is NOT a GPU allocation figure.  It measures total
                    system RAM in use by the OS, all processes, and any GPU
                    workloads combined.  It fires only on unified-memory
                    architectures (Intel integrated Arc, AMD APU) where the
                    GPU and CPU share one physical pool, and only when every
                    vendor-specific backend has already failed.
                    NOT meaningful on discrete-VRAM hardware.
                    MUST NOT be compared with or pooled alongside rows whose
                    source is "nvidia-smi", "rocm-smi", or "intel-level-zero".
                    See THREATS.md §13.

"unavailable"       No backend succeeded.  The companion gpu_mem_mb field is
                    None.  Never 0.0.

Comparability warning
---------------------
"nvidia-smi" and "unified-psutil" measure fundamentally different things.
An Nvidia discrete reading of 4096 MiB means 4 GiB of *VRAM* is in use.
A unified-psutil reading of 4096 MiB means 4 GiB of the *shared system pool*
is in use by everything combined.  Do not compare them without correction.
Filter result rows by gpu_mem_source before drawing any memory-pressure
conclusions.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, fields


@dataclass
class Telemetry:
    latency_ms: float
    ttft_ms: float
    tokens_in: int
    tokens_out: int
    mem_rss_mb: float
    gpu_mem_mb: float | None = None
    gpu_mem_source: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Telemetry":
        known = {f.name for f in fields(cls)}
        # Accept legacy rows written with "gpu_mem_method" before the rename.
        mapped = dict(d)
        if "gpu_mem_method" in mapped and "gpu_mem_source" not in mapped:
            mapped["gpu_mem_source"] = mapped.pop("gpu_mem_method")
        return cls(**{k: v for k, v in mapped.items() if k in known})


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# ── Legacy private helpers (kept for backward compatibility; tested directly) ──

def _read_nvidia_smi() -> tuple[float | None, str | None]:
    """Query nvidia-smi for discrete GPU memory usage.

    Returns (used_mb, "nvidia_smi") on success.
    Returns (None, "unavailable:<reason>") on any failure so that a missing
    nvidia-smi is never reported as 0 MB used.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip().split("\n")[0]), "nvidia_smi"
    except FileNotFoundError:
        return None, "unavailable:nvidia_smi_not_found"
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}"


def _read_unified_pool() -> tuple[float | None, str | None]:
    """Read system-wide used memory from /proc/meminfo for unified-memory hosts.

    On unified-memory systems (e.g. AMD Strix Halo LPDDR5X) the GPU and CPU
    share one physical pool; there is no separate VRAM counter.  /proc/meminfo
    MemTotal - MemAvailable is the in-use share of the shared pool.  This is
    not GPU-only memory — it includes OS, all processes, and GPU allocations.
    rocm-smi VRAM reporting for Strix Halo is not yet reliable (as of
    2026-09); this is the best available proxy.  The method field labels the
    measurement so analysis code can distinguish it from discrete-GPU readings.

    Returns (None, "unavailable:<reason>") on any failure.
    """
    try:
        info: dict[str, str] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
        total_kb = int(info["MemTotal"].split()[0])
        avail_kb = int(info["MemAvailable"].split()[0])
        used_mb = (total_kb - avail_kb) / 1024
        return round(used_mb, 1), "unified_sys_pool"
    except FileNotFoundError:
        return None, "unavailable:proc_meminfo_not_found"
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}"


# ── New vendor-aware query helpers ────────────────────────────────────────────
# Each returns (used_mib: float, source_literal: str) on success or
# (None, "unavailable") on any failure.  Never returns 0.0 on failure.

def _query_nvidia_smi() -> tuple[float | None, str]:
    """Discrete NVIDIA VRAM in use (MiB) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        val = float(out.decode().strip().split("\n")[0])
        return val, "nvidia-smi"
    except Exception:
        return None, "unavailable"


def _query_rocm_smi() -> tuple[float | None, str]:
    """AMD GPU VRAM Total Used Memory (MiB) via rocm-smi.

    On AMD Strix Halo (unified LPDDR5X) this is the GPU driver's allocation
    within the shared pool, not whole-system usage.
    """
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--noheader"],
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        for line in out.decode().splitlines():
            if "Total Used Memory" in line or "VRAM Total Used" in line:
                # e.g. "GPU[0]  : VRAM Total Used Memory (B): 536870912"
                _, _, tail = line.rpartition(":")
                used_bytes = int(tail.strip())
                return round(used_bytes / 1024 / 1024, 1), "rocm-smi"
        return None, "unavailable"
    except Exception:
        return None, "unavailable"


def _query_intel_level_zero() -> tuple[float | None, str]:
    """Intel GPU memory via Level Zero Sysman (zesMemoryGetState).

    Sums all memory modules reported by the first GPU device.
    On integrated Arc (unified pool) this is GPU-allocated shared DRAM.
    Falls through on any failure: library absent, API error, unsupported.
    """
    try:
        import ctypes
        import ctypes.util
        import sys as _sys

        if _sys.platform == "win32":
            try:
                _ze = ctypes.WinDLL("ze_loader.dll")
            except OSError:
                return None, "unavailable"
        else:
            lib = ctypes.util.find_library("ze_loader") or "libze_loader.so.1"
            try:
                _ze = ctypes.CDLL(lib)
            except OSError:
                return None, "unavailable"

        # ZES_ENABLE_SYSMAN=1 must be set before zeInit so the driver
        # initialises the Sysman layer.  Without it, zesDeviceEnumMemoryModules
        # returns ZE_RESULT_ERROR_UNINITIALIZED (0x78000001) even when the
        # device handle is valid.  The env var is read once at zeInit time;
        # setting it here (after the DLL is loaded) is sufficient on Windows.
        os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")

        # zeInit(ZE_INIT_FLAG_GPU_ONLY=0x1)
        _ze.zeInit.restype = ctypes.c_int
        _ze.zeInit.argtypes = [ctypes.c_uint32]
        if _ze.zeInit(0x1) != 0:
            return None, "unavailable"

        # zeDriverGet
        _ze.zeDriverGet.restype = ctypes.c_int
        _ze.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        n = ctypes.c_uint32(0)
        if _ze.zeDriverGet(ctypes.byref(n), None) != 0 or n.value == 0:
            return None, "unavailable"
        drivers = (ctypes.c_void_p * n.value)()
        _ze.zeDriverGet(ctypes.byref(n), drivers)

        # zeDeviceGet — first driver, first device
        _ze.zeDeviceGet.restype = ctypes.c_int
        _ze.zeDeviceGet.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        nd = ctypes.c_uint32(0)
        _ze.zeDeviceGet(drivers[0], ctypes.byref(nd), None)
        if nd.value == 0:
            return None, "unavailable"
        devs = (ctypes.c_void_p * nd.value)()
        _ze.zeDeviceGet(drivers[0], ctypes.byref(nd), devs)
        device = devs[0]

        # zesDeviceEnumMemoryModules
        _ze.zesDeviceEnumMemoryModules.restype = ctypes.c_int
        _ze.zesDeviceEnumMemoryModules.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
        ]
        nm = ctypes.c_uint32(0)
        if _ze.zesDeviceEnumMemoryModules(device, ctypes.byref(nm), None) != 0 or nm.value == 0:
            return None, "unavailable"
        mems = (ctypes.c_void_p * nm.value)()
        _ze.zesDeviceEnumMemoryModules(device, ctypes.byref(nm), mems)

        # zesMemoryGetState — struct layout (64-bit):
        #   stype  i32 @0  pNext ptr @8  health i32 @16  free u64 @24  size u64 @32
        class _ZesMemState(ctypes.Structure):
            _fields_ = [
                ("stype",   ctypes.c_int32),
                ("pNext",   ctypes.c_void_p),   # natural align adds 4-byte pad before this
                ("health",  ctypes.c_int32),
                ("free",    ctypes.c_uint64),
                ("size",    ctypes.c_uint64),
            ]

        _ze.zesMemoryGetState.restype = ctypes.c_int
        _ze.zesMemoryGetState.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZesMemState)]

        total_used_bytes = 0
        found = False
        for i in range(nm.value):
            st = _ZesMemState()
            st.stype = 0x3  # ZES_STRUCTURE_TYPE_MEM_STATE
            st.pNext = None
            if _ze.zesMemoryGetState(mems[i], ctypes.byref(st)) == 0:
                total_used_bytes += max(0, st.size - st.free)
                found = True

        if not found:
            return None, "unavailable"
        return round(total_used_bytes / 1024 / 1024, 1), "intel-level-zero"

    except Exception:
        return None, "unavailable"


def _query_intel_sysfs() -> tuple[float | None, str]:
    """Intel i915 debug-fs GEM object total on Linux (requires debugfs + read access).

    Reads /sys/kernel/debug/dri/<N>/i915_gem_objects for the first card with
    vendor 0x8086.  Falls through (returns unavailable) without root or when
    debugfs is not mounted.  On most production systems this will fall through.
    """
    try:
        import glob as _glob
        for vendor_path in _glob.glob("/sys/class/drm/card*/device/vendor"):
            try:
                vendor = open(vendor_path).read().strip()
            except OSError:
                continue
            if vendor != "0x8086":
                continue
            # Extract card number
            card_n = vendor_path.split("/card")[1].split("/")[0]
            gem_path = f"/sys/kernel/debug/dri/{card_n}/i915_gem_objects"
            try:
                text = open(gem_path).read()
            except OSError:
                continue
            # Parse "total gtt size: 12345678 bytes" or similar summary line
            for line in text.splitlines():
                lc = line.lower()
                if "total" in lc and "byte" in lc:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.isdigit():
                            return round(int(p) / 1024 / 1024, 1), "intel-sysfs"
        return None, "unavailable"
    except Exception:
        return None, "unavailable"


def _query_unified_psutil() -> tuple[float | None, str]:
    """Whole-system used memory (MiB) via psutil.

    On unified-memory architectures (Intel integrated Arc, AMD APU) the
    system pool IS the GPU pool; this is a whole-system proxy that includes
    OS, all processes, and GPU allocations combined.  Cross-platform.
    NOT meaningful on discrete-VRAM hardware.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        used_mb = (vm.total - vm.available) / 1024 / 1024
        return round(used_mb, 1), "unified-psutil"
    except Exception:
        return None, "unavailable"


# ── Public API ────────────────────────────────────────────────────────────────

def gpu_mem_mb(memory_architecture: str = "discrete") -> tuple[float | None, str]:
    """Sample GPU/memory usage, returning (used_mib, source_literal).

    Tries backends in order, falling through only on failure:
      a. nvidia-smi          → "nvidia-smi"       (discrete NVIDIA VRAM)
      b. rocm-smi            → "rocm-smi"         (AMD GPU VRAM)
      c. Intel Level Zero    → "intel-level-zero" (Intel GPU via Sysman)
      d. Intel sysfs         → "intel-sysfs"      (Linux i915 debugfs; rarely succeeds)
      e. psutil system pool  → "unified-psutil"   (unified arch only; whole-system)

    Returns (None, "unavailable") when no backend succeeds.
    Never returns 0.0 as a proxy for unavailability.

    memory_architecture:
      "unified"  — psutil system-pool fallback (step e) is enabled.
      "discrete" — psutil fallback is skipped; system RAM ≠ VRAM.
      anything else — returns (None, "unavailable:unknown_architecture:<value>")
                      immediately without attempting any backend.

    See module docstring for per-source comparability warnings.
    """
    if memory_architecture not in ("unified", "discrete"):
        return None, f"unavailable:unknown_architecture:{memory_architecture}"

    for query in (
        _query_nvidia_smi,
        _query_rocm_smi,
        _query_intel_level_zero,
        _query_intel_sysfs,
    ):
        val, src = query()
        if val is not None:
            return val, src

    if memory_architecture == "unified":
        val, src = _query_unified_psutil()
        if val is not None:
            return val, src

    return None, "unavailable"
