"""CPU topology discovery and empirical P / LP-E verification (spec §4).

Platform A is 4 Cougar Cove P-cores + 4 Darkmont LP-E cores, 8 logical CPUs, **no SMT**. The
``cpu-p`` and ``cpu-lpe`` execution targets depend on knowing which logical CPU is which.

Spec §4 is explicit: **do not trust the Windows ``EfficiencyClass`` ordering blindly.** So this
module treats the OS as a *hypothesis* and measurement as the authority:

1. Enumerate cores via ``GetLogicalProcessorInformationEx(RelationProcessorCore)`` and read
   ``EfficiencyClass`` per core.
2. Run a fixed single-thread integer+FP microbenchmark pinned to each logical CPU.
3. Cluster the scores into exactly two groups and require a clean separation matching the
   expected split, with the P-cluster faster.
4. Compare the measured partition against the ``EfficiencyClass`` partition and record whether
   they agree. **That comparison is spec §10 open question 4** - it is an output of this module,
   never an input to it.

The direction of ``EfficiencyClass`` is deliberately not hardcoded. Both interpretations are
tested and the result recorded, because assuming a direction would answer open question 4 by
assertion.

**Measurement and verdict are separate** (``AUDIT_LOG.md`` AF-006). :func:`measure_topology`
always produces a full :class:`TopologyResult` carrying every acceptance check it applied and
whether each held; :func:`verify_topology` is the thin wrapper that turns a refusal into an
exception. The split exists so the CLI can emit a manifest for a refused verification: a refusal
is a measurement of real silicon, and spec §9.2 ("no number without a ``run_id``") applies to it
just as much as to a pass.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import platform
import struct
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from seam.config import ResolvedConfig, load_platform_config
from seam.errors import (
    ConfigError,
    TopologyNotVerifiedError,
    TopologyVerificationError,
)
from seam.jsonlog import log_event, utc_now_iso
from seam.powerstate import (
    assert_pinned_for_committed_result,
    capture_power_state,
    check_pinned_conditions,
    manifest_power_state,
)

__all__ = [
    "CoreInfo",
    "CpuTarget",
    "KernelSpec",
    "TopologyResult",
    "VerifiedTopology",
    "affinity_for",
    "build_kernel",
    "enumerate_cores",
    "load_verified_topology",
    "measure_topology",
    "verify_topology",
]

CpuTarget = Literal["cpu-p", "cpu-lpe"]

#: Verdict recorded by :func:`measure_topology`. ``"refused"`` means the measurement completed but
#: at least one acceptance criterion did not hold, so the mapping is NOT established.
TopologyVerdict = Literal["pass", "refused"]

#: ``LOGICAL_PROCESSOR_RELATIONSHIP.RelationProcessorCore``.
_RELATION_PROCESSOR_CORE: Final = 0
#: ``ERROR_INSUFFICIENT_BUFFER`` - expected from the sizing call.
_ERROR_INSUFFICIENT_BUFFER: Final = 122

# Byte offsets within SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX when the union holds a
# PROCESSOR_RELATIONSHIP. Parsed by explicit offset rather than via ctypes.Union because the
# trailing GroupMask[] is a variable-length array, which ctypes cannot describe directly.
#
#   DWORD Relationship          @ 0
#   DWORD Size                  @ 4
#   BYTE  Flags                 @ 8
#   BYTE  EfficiencyClass       @ 9
#   BYTE  Reserved[20]          @ 10
#   WORD  GroupCount            @ 30
#   GROUP_AFFINITY GroupMask[]  @ 32   (ULONG_PTR Mask @ +0, WORD Group @ +8, WORD Reserved[3])
_OFF_RELATIONSHIP: Final = 0
_OFF_SIZE: Final = 4
_OFF_FLAGS: Final = 8
_OFF_EFFICIENCY_CLASS: Final = 9
_OFF_GROUP_COUNT: Final = 30
_OFF_GROUP_MASK: Final = 32
_SIZEOF_GROUP_AFFINITY: Final = 16

#: ``PROCESSOR_RELATIONSHIP.Flags`` bit indicating the core has more than one logical processor
#: (i.e. SMT is present on that core). Platform A must report 0.
_LTP_PC_SMT: Final = 0x1


@dataclass(frozen=True, slots=True)
class CoreInfo:
    """One physical core as reported by Windows."""

    core_index: int
    efficiency_class: int
    logical_cpus: tuple[int, ...]
    group: int
    smt: bool


@dataclass(slots=True)
class TopologyResult:
    """Full result of an empirical verification run, whether it passed or refused.

    Serialised verbatim into ``raw/<run_id>/summary.json`` so every field below is traceable.

    ``p_cpus`` and ``lpe_cpus`` name the measured fast and slow clusters. **On a refused run they
    are a clustering of the scores, not a verified mapping** - ``verdict`` is the field that says
    which, and nothing may consume them as a mapping unless ``verdict == "pass"``.
    """

    timestamp_utc: str
    n_logical_cpus: int
    cores: list[dict[str, Any]]
    efficiency_class_map: dict[int, int]
    scores: dict[int, float]
    p_cpus: list[int]
    lpe_cpus: list[int]
    cluster_separation_ratio: float
    p_cluster_cv: float
    lpe_cluster_cv: float

    #: ``"pass"`` or ``"refused"``.
    verdict: TopologyVerdict = "refused"
    #: One entry per acceptance criterion that did not hold. Empty iff ``verdict == "pass"``.
    refusal_reasons: list[str] = field(default_factory=list)

    # --- Spec §10 open question 4 -------------------------------------------------------------
    #: True iff the EfficiencyClass partition equals the measured partition AND a higher
    #: EfficiencyClass value corresponds to the faster (P) cluster, which is the documented
    #: Windows interpretation.
    efficiency_class_ordering_matched: bool = False
    #: True iff the partitions agree as *sets*, ignoring which value means "fast". Separated from
    #: the above so a correct grouping with an inverted sense is not reported as a flat failure.
    efficiency_class_partition_matched: bool = False
    #: ``"higher_is_faster"``, ``"lower_is_faster"``, or ``"undetermined"``.
    efficiency_class_direction: str = "undetermined"

    benchmark: dict[str, Any] = field(default_factory=dict)
    host: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerifiedTopology:
    """A verified mapping read back from platform config."""

    p_cpus: tuple[int, ...]
    lpe_cpus: tuple[int, ...]
    verified: bool
    run_id: str | None


# ==================================================================================================
# Windows core enumeration
# ==================================================================================================


def enumerate_cores() -> list[CoreInfo]:
    """Enumerate physical cores via ``GetLogicalProcessorInformationEx``.

    Returns:
        One :class:`CoreInfo` per physical core, in the order Windows reports them.

    Raises:
        TopologyVerificationError: If not running on Windows, or if the Win32 call fails. Never
            falls back to ``os.cpu_count()``: a guessed topology is worse than no topology,
            because it looks like a measurement.
    """
    if sys.platform != "win32":
        raise TopologyVerificationError(
            f"GetLogicalProcessorInformationEx is Windows-only; running on {sys.platform!r}. "
            f"Platform A measurement must run on native Windows (spec §3)."
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    length = ctypes.c_ulong(0)

    # First call sizes the buffer and is *expected* to fail with ERROR_INSUFFICIENT_BUFFER.
    ok = kernel32.GetLogicalProcessorInformationEx(
        ctypes.c_ulong(_RELATION_PROCESSOR_CORE), None, ctypes.byref(length)
    )
    if ok:
        raise TopologyVerificationError(
            "GetLogicalProcessorInformationEx unexpectedly succeeded with a NULL buffer"
        )
    err = ctypes.get_last_error()
    if err != _ERROR_INSUFFICIENT_BUFFER:
        raise TopologyVerificationError(
            f"GetLogicalProcessorInformationEx sizing call failed with Win32 error {err} "
            f"(expected {_ERROR_INSUFFICIENT_BUFFER} ERROR_INSUFFICIENT_BUFFER)"
        )

    buffer = ctypes.create_string_buffer(length.value)
    ok = kernel32.GetLogicalProcessorInformationEx(
        ctypes.c_ulong(_RELATION_PROCESSOR_CORE), buffer, ctypes.byref(length)
    )
    if not ok:
        raise TopologyVerificationError(
            f"GetLogicalProcessorInformationEx failed with Win32 error {ctypes.get_last_error()}"
        )

    return _parse_processor_core_records(buffer.raw[: length.value])


def _parse_processor_core_records(raw: bytes) -> list[CoreInfo]:
    """Parse a ``SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`` buffer of processor-core records.

    Separated from the Win32 call so it can be unit-tested with a synthetic buffer on any OS.

    Raises:
        TopologyVerificationError: If the buffer is malformed.
    """
    cores: list[CoreInfo] = []
    offset = 0
    core_index = 0

    while offset < len(raw):
        if offset + _OFF_GROUP_MASK > len(raw):
            raise TopologyVerificationError(
                f"truncated processor-information record at offset {offset}"
            )

        relationship = struct.unpack_from("<I", raw, offset + _OFF_RELATIONSHIP)[0]
        size = struct.unpack_from("<I", raw, offset + _OFF_SIZE)[0]

        if size == 0:
            raise TopologyVerificationError(
                f"processor-information record at offset {offset} declares Size=0; "
                f"refusing to loop forever"
            )
        if offset + size > len(raw):
            raise TopologyVerificationError(
                f"record at offset {offset} declares Size={size} which overruns the "
                f"{len(raw)}-byte buffer"
            )

        # We asked for RelationProcessorCore only, but the record is skipped rather than trusted
        # if the kernel returns anything else.
        if relationship != _RELATION_PROCESSOR_CORE:
            log_event(
                "topology.unexpected_relationship",
                severity="warning",
                message="skipping a non-processor-core record",
                relationship=relationship,
                offset=offset,
            )
            offset += size
            continue

        flags = raw[offset + _OFF_FLAGS]
        efficiency_class = raw[offset + _OFF_EFFICIENCY_CLASS]
        group_count = struct.unpack_from("<H", raw, offset + _OFF_GROUP_COUNT)[0]

        logical_cpus: list[int] = []
        group_id = 0
        for group_ix in range(group_count):
            mask_offset = offset + _OFF_GROUP_MASK + group_ix * _SIZEOF_GROUP_AFFINITY
            if mask_offset + _SIZEOF_GROUP_AFFINITY > offset + size:
                raise TopologyVerificationError(
                    f"GroupMask[{group_ix}] overruns record at offset {offset}"
                )
            # KAFFINITY is ULONG_PTR: 8 bytes on x64.
            mask = struct.unpack_from("<Q", raw, mask_offset)[0]
            group_id = struct.unpack_from("<H", raw, mask_offset + 8)[0]
            # Processor groups cap at 64 logical CPUs, so a 64-bit mask is complete.
            logical_cpus.extend(bit for bit in range(64) if mask & (1 << bit))

        cores.append(
            CoreInfo(
                core_index=core_index,
                efficiency_class=efficiency_class,
                logical_cpus=tuple(sorted(logical_cpus)),
                group=group_id,
                smt=bool(flags & _LTP_PC_SMT),
            )
        )
        core_index += 1
        offset += size

    if not cores:
        raise TopologyVerificationError("no processor-core records returned")
    return cores


# ==================================================================================================
# Microbenchmark
# ==================================================================================================


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """One fully-parameterised microbenchmark kernel.

    Spec §4 prescribes "a fixed single-thread integer+FP benchmark" but not its implementation, and
    which implementation runs is a measurable property of a run - so the kernel is selected by name
    from config (``topology.verification.kernel``) and its name is recorded as the manifest's
    ``workload.benchmark``. Changing the instrument therefore changes ``config_hash``, and cannot
    happen invisibly.
    """

    name: str
    #: Abstract units of work per trial. Only ratios between CPUs are meaningful, so the unit needs
    #: to be consistent across CPUs, not comparable across kernels.
    work_units_per_trial: int
    #: Runs one trial. Must be deterministic, single-threaded, and allocation-free in its hot path.
    run: Callable[[], object]
    #: The resolved parameters, recorded in the summary so a score can be reproduced.
    params: dict[str, Any]


def _python_intfp_v1(integer_iterations: int, float_iterations: int) -> tuple[int, float]:
    """Fixed integer + floating-point work in pure CPython.

    Both accumulators are returned so the interpreter cannot optimise the loops away.

    The kernel is deliberately dependency-free and branch-light: each iteration depends on the
    previous accumulator, so it measures single-thread latency-bound throughput rather than the
    memory system.

    Every iteration is a handful of CPython bytecodes, so a large share of its cost is interpreter
    dispatch and reference counting rather than the arithmetic itself. That was the standing
    suspicion for why two earlier runs measured only 1.12x and 1.19x separation (AUDIT_LOG.md,
    2026-07-29). It was **not** the cause: under MACHINE.md's pinned power plan the same kernel
    separates the clusters by 1.309x (run `0d7c607b-b80b-4563-a5cc-00fa34d51fd8`), and absolute
    throughput is ~2.4x higher. The compression was the Balanced power plan and the
    "Best power efficiency" overlay clamping both core types, not the instrument.
    """
    acc_i = 1
    for i in range(integer_iterations):
        acc_i = (acc_i * 1_103_515_245 + 12_345 + i) & 0x7FFF_FFFF
        acc_i ^= acc_i >> 7

    acc_f = 1.0
    for _ in range(float_iterations):
        acc_f = acc_f * 1.0000001 + 0.5
        acc_f -= 0.25

    return acc_i, acc_f


def _build_python_intfp_v1(params: Mapping[str, Any]) -> KernelSpec:
    integer_iterations = int(params["integer_iterations"])
    float_iterations = int(params["float_iterations"])
    return KernelSpec(
        name="python_intfp_v1",
        work_units_per_trial=integer_iterations + float_iterations,
        run=lambda: _python_intfp_v1(integer_iterations, float_iterations),
        params={
            "integer_iterations": integer_iterations,
            "float_iterations": float_iterations,
            "work_unit": "loop_iteration",
        },
    )


#: Kernel name -> builder. A kernel must be registered here to be selectable, so config cannot name
#: an instrument that does not exist.
_KERNEL_BUILDERS: Final[dict[str, Callable[[Mapping[str, Any]], KernelSpec]]] = {
    "python_intfp_v1": _build_python_intfp_v1,
}


def build_kernel(name: str, params: Mapping[str, Any]) -> KernelSpec:
    """Build the named kernel from its config parameters.

    Raises:
        ConfigError: If the kernel is not registered, or a parameter it needs is missing. Never
            falls back to another kernel: silently measuring with a different instrument than the
            config names would make every score untraceable.
    """
    builder = _KERNEL_BUILDERS.get(name)
    if builder is None:
        raise ConfigError(
            f"unknown topology kernel {name!r}; registered kernels are "
            f"{sorted(_KERNEL_BUILDERS)}. Refusing to substitute a different instrument."
        )
    try:
        return builder(params)
    except KeyError as exc:
        raise ConfigError(
            f"topology kernel {name!r} requires config parameter {exc.args[0]!r}, which is absent "
            f"from topology.verification"
        ) from exc


def _bench_one_cpu(
    cpu: int,
    *,
    kernel: KernelSpec,
    repeats: int,
    warmup_repeats: int,
) -> float:
    """Pin to ``cpu``, run the kernel, and return a score in work-units per second.

    The score is derived from the **minimum** elapsed time across repeats, not the mean.
    Interference from other processes can only ever make a trial slower, so the minimum is the
    cleanest estimate of the core's capability.

    Raises:
        TopologyVerificationError: If the CPU cannot be pinned. Never proceeds unpinned, because
            an unpinned trial silently measures whichever core the scheduler chose.
    """
    import psutil

    process = psutil.Process()
    original_affinity = process.cpu_affinity()

    try:
        try:
            process.cpu_affinity([cpu])
        except (OSError, ValueError, psutil.AccessDenied) as exc:
            raise TopologyVerificationError(
                f"could not pin to logical CPU {cpu}: {exc}. Refusing to benchmark unpinned."
            ) from exc

        actual = process.cpu_affinity()
        if set(actual) != {cpu}:
            raise TopologyVerificationError(
                f"affinity for logical CPU {cpu} did not take effect; process reports {actual}"
            )

        best_ns: int | None = None
        for repeat in range(warmup_repeats + repeats):
            start = time.perf_counter_ns()
            kernel.run()
            elapsed = time.perf_counter_ns() - start
            if repeat < warmup_repeats:
                continue
            if best_ns is None or elapsed < best_ns:
                best_ns = elapsed

        if best_ns is None or best_ns <= 0:
            raise TopologyVerificationError(
                f"benchmark on logical CPU {cpu} produced no usable timing "
                f"(best_ns={best_ns}); check repeats_per_cpu > 0"
            )

        return kernel.work_units_per_trial / (best_ns / 1e9)
    finally:
        # Restore affinity even on failure, so a raised error does not leave the process pinned to
        # one core for the remainder of the session.
        try:
            process.cpu_affinity(original_affinity)
        except (OSError, ValueError) as exc:
            log_event(
                "topology.affinity_restore_failed",
                severity="error",
                message="could not restore original CPU affinity",
                original_affinity=original_affinity,
                error=str(exc),
            )


# ==================================================================================================
# Clustering
# ==================================================================================================


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _cv(values: list[float]) -> float:
    """Coefficient of variation (population). Zero for a single value."""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    if mu == 0:
        raise TopologyVerificationError("cannot compute CV for a zero-mean cluster")
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mu


def _split_into_two_clusters(scores: dict[int, float]) -> tuple[list[int], list[int]]:
    """Partition logical CPUs into a fast and a slow cluster.

    For one-dimensional data the optimal 2-means partition is always contiguous in sorted order,
    so every split point is evaluated exactly and the lowest total within-cluster sum of squares
    wins. No iteration, no seeding, no local minimum - the result is deterministic, which matters
    for a value that gets committed to config.

    Returns:
        ``(fast_cpus, slow_cpus)``, each sorted ascending by CPU index.
    """
    if len(scores) < 2:
        raise TopologyVerificationError(
            f"need at least 2 logical CPUs to find two clusters, got {len(scores)}"
        )

    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    values = [value for _, value in ordered]

    best_split: int | None = None
    best_cost = float("inf")
    for split in range(1, len(values)):
        low, high = values[:split], values[split:]
        cost = sum((v - _mean(low)) ** 2 for v in low) + sum((v - _mean(high)) ** 2 for v in high)
        if cost < best_cost:
            best_cost = cost
            best_split = split

    if best_split is None:
        raise TopologyVerificationError("failed to find a cluster split point")

    slow = sorted(cpu for cpu, _ in ordered[:best_split])
    fast = sorted(cpu for cpu, _ in ordered[best_split:])
    return fast, slow


def _evaluate_efficiency_class_agreement(
    cores: list[CoreInfo],
    p_cpus: list[int],
    lpe_cpus: list[int],
) -> tuple[bool, bool, str]:
    """Compare the ``EfficiencyClass`` partition against the measured partition.

    **This is spec §10 open question 4.** The direction of ``EfficiencyClass`` is not assumed:
    both senses are tested and the observed one is reported.

    Returns:
        ``(ordering_matched, partition_matched, direction)``.
    """
    classes = sorted({core.efficiency_class for core in cores})
    if len(classes) != 2:
        log_event(
            "topology.efficiency_class_not_bimodal",
            severity="warning",
            message=(
                "Windows did not report exactly two distinct EfficiencyClass values; "
                "open question 4 cannot be answered as a simple match"
            ),
            distinct_efficiency_classes=classes,
        )
        return False, False, "undetermined"

    low_class, high_class = classes
    cpus_low = {
        cpu for core in cores if core.efficiency_class == low_class for cpu in core.logical_cpus
    }
    cpus_high = {
        cpu for core in cores if core.efficiency_class == high_class for cpu in core.logical_cpus
    }

    measured_p, measured_lpe = set(p_cpus), set(lpe_cpus)

    # Documented Windows interpretation: a larger EfficiencyClass means greater performance.
    higher_is_faster = cpus_high == measured_p and cpus_low == measured_lpe
    lower_is_faster = cpus_low == measured_p and cpus_high == measured_lpe

    partition_matched = higher_is_faster or lower_is_faster
    if higher_is_faster:
        direction = "higher_is_faster"
    elif lower_is_faster:
        direction = "lower_is_faster"
    else:
        direction = "undetermined"

    return higher_is_faster, partition_matched, direction


# ==================================================================================================
# Verification
# ==================================================================================================


def measure_topology(config: ResolvedConfig) -> TopologyResult:
    """Measure the P / LP-E split and record whether it meets the §4 acceptance criteria.

    Every acceptance criterion is evaluated and recorded rather than short-circuited, so a refused
    run still reports its separation ratio, both cluster CVs, and the ``EfficiencyClass``
    comparison. The earlier refused runs could report none of those, because the first failing
    check raised (``AUDIT_LOG.md`` AF-006).

    Returns:
        A :class:`TopologyResult` whose ``verdict`` is ``"pass"`` only if every criterion held.

    Raises:
        TopologyVerificationError: Only for conditions that make a *measurement* impossible - not
            running on Windows, core enumeration failing, a CPU that cannot be pinned, or a
            platform that is not the declared part. Those are broken instruments, not verdicts.
    """
    params = config.require("topology.verification")
    expected = config.require("topology.expected")

    kernel = build_kernel(str(params["kernel"]), params)
    repeats = int(params["repeats_per_cpu"])
    warmup_repeats = int(params["warmup_repeats"])
    min_separation = float(params["min_cluster_separation_ratio"])
    max_within_cv = float(params["max_within_cluster_cv"])
    require_expected = bool(params["require_expected_split"])

    cores = enumerate_cores()
    logical_cpus = sorted(cpu for core in cores for cpu in core.logical_cpus)
    efficiency_class_map = {
        cpu: core.efficiency_class for core in cores for cpu in core.logical_cpus
    }

    log_event(
        "topology.enumerated",
        message=(
            f"Windows reports {len(cores)} physical core(s), {len(logical_cpus)} logical CPU(s)"
        ),
        n_cores=len(cores),
        n_logical_cpus=len(logical_cpus),
        efficiency_class_map=efficiency_class_map,
        smt_flags={core.core_index: core.smt for core in cores},
    )

    expected_logical = int(expected["n_logical_cpus"])
    if len(logical_cpus) != expected_logical:
        raise TopologyVerificationError(
            f"expected {expected_logical} logical CPUs (spec §1: 8, no SMT) but Windows reports "
            f"{len(logical_cpus)}: {logical_cpus}. Refusing to verify a topology that does not "
            f"match the declared platform."
        )

    if not bool(expected["smt"]) and any(core.smt for core in cores):
        raise TopologyVerificationError(
            "platform config declares smt: false but Windows reports at least one SMT core; "
            "the platform identity or the config is wrong"
        )

    log_event(
        "topology.kernel_selected",
        message=(
            f"microbenchmark kernel {kernel.name!r}, "
            f"{kernel.work_units_per_trial} work-units per trial"
        ),
        kernel=kernel.name,
        work_units_per_trial=kernel.work_units_per_trial,
        params=kernel.params,
    )

    scores: dict[int, float] = {}
    for cpu in logical_cpus:
        score = _bench_one_cpu(
            cpu,
            kernel=kernel,
            repeats=repeats,
            warmup_repeats=warmup_repeats,
        )
        scores[cpu] = score
        log_event(
            "topology.cpu_benchmarked",
            message=f"logical CPU {cpu}: {score / 1e6:.3f} M work-units/s",
            cpu=cpu,
            score_units_per_s=score,
            efficiency_class=efficiency_class_map[cpu],
        )

    fast_cpus, slow_cpus = _split_into_two_clusters(scores)
    fast_values = [scores[cpu] for cpu in fast_cpus]
    slow_values = [scores[cpu] for cpu in slow_cpus]

    slow_mean, fast_mean = _mean(slow_values), _mean(fast_values)
    if slow_mean <= 0:
        raise TopologyVerificationError("slow cluster has non-positive mean score")
    separation = fast_mean / slow_mean

    fast_cv, slow_cv = _cv(fast_values), _cv(slow_values)

    # The faster cluster is the P cluster by definition of "performance core" - but only if the
    # criteria below hold, which is what `verdict` records.
    p_cpus, lpe_cpus = fast_cpus, slow_cpus

    refusal_reasons: list[str] = []

    if separation < min_separation:
        refusal_reasons.append(
            f"clusters do not separate cleanly: ratio {separation:.3f} < required "
            f"{min_separation:.3f}. The two core types are not distinguishable by this benchmark "
            f"at this threshold, so the P/LP-E mapping is NOT established."
        )

    if fast_cv > max_within_cv or slow_cv > max_within_cv:
        refusal_reasons.append(
            f"within-cluster spread too high (fast CV {fast_cv:.4f}, slow CV {slow_cv:.4f}, "
            f"limit {max_within_cv:.4f}); clusters are not clean. Likely background load - "
            f"quiesce the machine and re-run."
        )

    if require_expected:
        expected_p = int(expected["n_p_cores"])
        expected_lpe = int(expected["n_lpe_cores"])
        if len(p_cpus) != expected_p or len(lpe_cpus) != expected_lpe:
            refusal_reasons.append(
                f"measured split {len(p_cpus)}P/{len(lpe_cpus)}LP-E does not match the expected "
                f"{expected_p}P/{expected_lpe}LP-E. Either the platform is not the declared part "
                f"or the benchmark is not discriminating."
            )

    ordering_matched, partition_matched, direction = _evaluate_efficiency_class_agreement(
        cores, p_cpus, lpe_cpus
    )

    verdict: TopologyVerdict = "refused" if refusal_reasons else "pass"
    log_event(
        "topology.verdict",
        severity="info" if verdict == "pass" else "warning",
        message=(
            f"verification {verdict.upper()}: separation {separation:.3f}x "
            f"(required {min_separation:.3f}x), CV fast {fast_cv:.4f} / slow {slow_cv:.4f}"
        ),
        verdict=verdict,
        refusal_reasons=refusal_reasons,
        cluster_separation_ratio=separation,
        min_cluster_separation_ratio=min_separation,
        p_cluster_cv=fast_cv,
        lpe_cluster_cv=slow_cv,
        scores={cpu: scores[cpu] for cpu in logical_cpus},
        fast_cpus=p_cpus,
        slow_cpus=lpe_cpus,
    )

    log_event(
        "topology.open_question_4",
        severity="info" if ordering_matched else "warning",
        message=(
            "EfficiencyClass ordering matched the measured P/LP-E split"
            if ordering_matched
            else "EfficiencyClass ordering did NOT match the measured split as documented"
        ),
        efficiency_class_ordering_matched=ordering_matched,
        efficiency_class_partition_matched=partition_matched,
        efficiency_class_direction=direction,
        efficiency_class_map=efficiency_class_map,
        measured_p_cpus=p_cpus,
        measured_lpe_cpus=lpe_cpus,
    )

    return TopologyResult(
        timestamp_utc=utc_now_iso(),
        n_logical_cpus=len(logical_cpus),
        cores=[asdict(core) for core in cores],
        efficiency_class_map=efficiency_class_map,
        scores=scores,
        p_cpus=p_cpus,
        lpe_cpus=lpe_cpus,
        cluster_separation_ratio=separation,
        p_cluster_cv=fast_cv,
        lpe_cluster_cv=slow_cv,
        efficiency_class_ordering_matched=ordering_matched,
        efficiency_class_partition_matched=partition_matched,
        efficiency_class_direction=direction,
        verdict=verdict,
        refusal_reasons=refusal_reasons,
        benchmark={
            "kernel": kernel.name,
            "work_units_per_trial": kernel.work_units_per_trial,
            **kernel.params,
            "repeats_per_cpu": repeats,
            "warmup_repeats": warmup_repeats,
            "score_estimator": "min_elapsed_over_repeats",
            "min_cluster_separation_ratio": min_separation,
            "max_within_cluster_cv": max_within_cv,
        },
        host={
            "platform": platform.platform(),
            "python": sys.version,
            "processor": platform.processor(),
        },
    )


def verify_topology(config: ResolvedConfig) -> TopologyResult:
    """Empirically determine the P / LP-E split, raising unless it is established (spec §4).

    Raises:
        TopologyVerificationError: If the clusters do not separate cleanly, do not match the
            expected split, or the P-cluster is not the faster one. Failure is a hard error, not a
            warning - a caller must not proceed on an ambiguous mapping. Callers that need the
            measurement regardless of the verdict use :func:`measure_topology` and inspect
            ``verdict`` themselves.
    """
    result = measure_topology(config)
    if result.verdict != "pass":
        raise TopologyVerificationError(
            "topology verification REFUSED: "
            + " ".join(result.refusal_reasons)
            + f" Scores: {result.scores}"
        )
    return result


# ==================================================================================================
# Reading and asserting the verified mapping
# ==================================================================================================


def load_verified_topology(config: ResolvedConfig) -> VerifiedTopology:
    """Read the verified mapping from platform config.

    Raises:
        TopologyNotVerifiedError: If ``topology.verified`` is falsey or the CPU lists are absent.
        ConfigError: If the mapping is present but internally inconsistent.
    """
    verified = bool(config.get("topology.verified", False))
    p_raw = config.get("topology.p_cpus")
    lpe_raw = config.get("topology.lpe_cpus")

    if not verified or p_raw is None or lpe_raw is None:
        raise TopologyNotVerifiedError(
            "platform config carries no verified P/LP-E mapping "
            f"(verified={verified}, p_cpus={p_raw}, lpe_cpus={lpe_raw}). "
            "Run `python -m seam.topology verify --write` on the target hardware. "
            "Spec §4 requires the mapping to be verified empirically and asserted on every run; "
            "guessing it would silently answer open question 4 with an assumption."
        )

    p_cpus = tuple(int(cpu) for cpu in p_raw)
    lpe_cpus = tuple(int(cpu) for cpu in lpe_raw)

    overlap = set(p_cpus) & set(lpe_cpus)
    if overlap:
        raise ConfigError(f"p_cpus and lpe_cpus overlap on {sorted(overlap)}")
    if len(set(p_cpus)) != len(p_cpus) or len(set(lpe_cpus)) != len(lpe_cpus):
        raise ConfigError(f"duplicate CPU indices in p_cpus={p_cpus} or lpe_cpus={lpe_cpus}")

    expected_total = int(config.require("topology.expected")["n_logical_cpus"])
    if len(p_cpus) + len(lpe_cpus) != expected_total:
        raise ConfigError(
            f"verified mapping covers {len(p_cpus) + len(lpe_cpus)} logical CPUs but the platform "
            f"declares {expected_total}"
        )

    return VerifiedTopology(
        p_cpus=p_cpus,
        lpe_cpus=lpe_cpus,
        verified=verified,
        run_id=config.get("topology.measured.run_id"),
    )


def affinity_for(
    target: CpuTarget,
    config: ResolvedConfig,
    *,
    allow_unverified: bool = False,
) -> list[int]:
    """Return the logical-CPU list for ``target`` (spec §4).

    Args:
        target: ``"cpu-p"`` or ``"cpu-lpe"``.
        config: Resolved platform config carrying the verified mapping.
        allow_unverified: Escape hatch for development on an unverified machine. Using it emits a
            ``topology.unverified_waiver`` event - spec §9.6 forbids a silent fallback, so the
            waiver is auditable and cannot be mistaken for a verified run.

    Raises:
        TopologyNotVerifiedError: If the mapping is unverified and ``allow_unverified`` is False.
        ValueError: If ``target`` is not a CPU target.
    """
    if target not in ("cpu-p", "cpu-lpe"):
        raise ValueError(
            f"affinity_for accepts 'cpu-p' or 'cpu-lpe', got {target!r}. "
            f"The 'igpu', 'npu', and 'cloud' targets have no CPU affinity."
        )

    try:
        verified_topology = load_verified_topology(config)
    except TopologyNotVerifiedError as exc:
        if not allow_unverified:
            raise
        expected = config.require("topology.expected")
        log_event(
            "topology.unverified_waiver",
            severity="warning",
            message=(
                "returning an UNVERIFIED affinity list because allow_unverified=True; "
                "this run's cpu-p / cpu-lpe attribution is NOT trustworthy and must not be "
                "used for any reported number"
            ),
            target=target,
            expected_n_p_cores=expected["n_p_cores"],
            expected_n_lpe_cores=expected["n_lpe_cores"],
        )
        raise TopologyNotVerifiedError(
            "allow_unverified=True was requested, but no fallback mapping is provided. "
            "Spec §4 offers no defensible default ordering: the whole point of open question 4 "
            "is that EfficiencyClass ordering may not match measurement. Run verification."
        ) from exc

    return list(verified_topology.p_cpus if target == "cpu-p" else verified_topology.lpe_cpus)


# ==================================================================================================
# CLI
# ==================================================================================================


def _write_verified_topology(
    yaml_path: Path,
    result: TopologyResult,
    *,
    run_id: str,
) -> None:
    """Write the measured mapping back into the platform YAML, preserving comments.

    Uses ``ruamel.yaml`` round-trip mode. PyYAML is deliberately not used for writing: it discards
    comments, and this file's comments carry the provenance classification that makes it
    auditable.

    The CPU lists are written **flow style** (``p_cpus: [0, 1, 2, 3]``) rather than as block
    sequences. That is not cosmetic. The comment introducing ``topology.expected`` is attached to
    the preceding ``lpe_cpus`` key, so replacing that key's scalar ``null`` with a *block* sequence
    emits the comment between the key and its items - leaving the text "Hypothesis, from vendor
    documentation. NOT evidence" sitting on top of the measured LP-E CPU list, which inverts its
    meaning. A flow sequence stays on the key's own line and the comment keeps its place.

    Raises:
        ConfigError: If ``ruamel.yaml`` is unavailable. Not silently downgraded to PyYAML, because
            that would destroy the provenance annotations without saying so.
    """
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedSeq
    except ImportError as exc:
        raise ConfigError(
            "writing verified topology requires ruamel.yaml for comment-preserving round-trip "
            "(`pip install ruamel.yaml`). Refusing to fall back to PyYAML, which would strip the "
            "provenance comments from the platform config."
        ) from exc

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    # Match the committed file's block-sequence indentation, so writing the mapping does not
    # reindent every unrelated list in the document.
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    # Emit an explicit `null` rather than an empty value. This file's editing rules give `null` a
    # meaning - "not yet measured", never to be replaced with a plausible number (AMENDMENTS.md
    # AM-006) - and a blank value cannot be distinguished from a field someone forgot to fill in.
    yaml_rt.representer.add_representer(
        type(None),
        lambda representer, _data: representer.represent_scalar("tag:yaml.org,2002:null", "null"),
    )
    with yaml_path.open("r", encoding="utf-8") as handle:
        document = yaml_rt.load(handle)

    def _inline(cpus: list[int]) -> CommentedSeq:
        seq = CommentedSeq(cpus)
        seq.fa.set_flow_style()
        return seq

    topology = document["topology"]
    topology["verified"] = True
    topology["p_cpus"] = _inline(result.p_cpus)
    topology["lpe_cpus"] = _inline(result.lpe_cpus)

    measured = topology["measured"]
    measured["run_id"] = run_id
    measured["timestamp_utc"] = result.timestamp_utc
    measured["kernel"] = result.benchmark["kernel"]
    measured["efficiency_class_map"] = {
        int(k): int(v) for k, v in result.efficiency_class_map.items()
    }
    measured["scores"] = {int(k): round(v, 3) for k, v in result.scores.items()}
    measured["cluster_separation_ratio"] = round(result.cluster_separation_ratio, 4)
    measured["p_cluster_cv"] = round(result.p_cluster_cv, 4)
    measured["lpe_cluster_cv"] = round(result.lpe_cluster_cv, 4)
    measured["efficiency_class_ordering_matched"] = result.efficiency_class_ordering_matched

    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml_rt.dump(document, handle)

    log_event(
        "topology.config_written",
        message=f"wrote verified topology to {yaml_path}",
        path=str(yaml_path),
        run_id=run_id,
        p_cpus=result.p_cpus,
        lpe_cpus=result.lpe_cpus,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m seam.topology verify [--write] [--allow-dirty]``."""
    parser = argparse.ArgumentParser(
        prog="python -m seam.topology",
        description="Empirically verify the Platform A P / LP-E core split (spec §4).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="run the microbenchmark and cluster the results")
    verify.add_argument("--platform-id", default="aipc-c1")
    verify.add_argument(
        "--write",
        action="store_true",
        help="persist the verified mapping into the platform YAML",
    )
    verify.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dirty git tree; recorded in the manifest",
    )

    show = subparsers.add_parser("show", help="print the enumerated cores without benchmarking")
    show.add_argument("--platform-id", default="aipc-c1")

    args = parser.parse_args(argv)

    # Imported here rather than at module scope to keep `seam.manifest` and `seam.topology` free of
    # a circular import: manifest.py reads topology, and only the CLI needs to emit a manifest.
    from seam.gitinfo import repo_root
    from seam.manifest import emit

    root = repo_root(Path(__file__).parent)
    config = load_platform_config(args.platform_id, repo_root=root)

    if args.command == "show":
        for core in enumerate_cores():
            log_event(
                "topology.core",
                message=(
                    f"core {core.core_index}: EfficiencyClass={core.efficiency_class} "
                    f"logical_cpus={list(core.logical_cpus)} smt={core.smt}"
                ),
                **asdict(core),
            )
        return 0

    # Captured BEFORE the benchmark so the manifest records the conditions the scores were taken
    # under, and so a session outside MACHINE.md's pinned conditions is visible in the artifact
    # rather than only in someone's memory (AUDIT_LOG.md AF-005).
    power_state = capture_power_state()
    power_cfg = config.get("power") or {}
    # Prefer profile-scoped assert (§3.7); fall back to legacy power.pinned for older configs.
    if power_cfg.get("profiles") and power_cfg.get("class_profile_map"):
        from seam.powerstate import assert_profile

        profile_assertion = assert_profile("topology_verify", power_state, power_cfg=power_cfg)
        pinned_deviations = list(profile_assertion.deviations)
    else:
        pinned_deviations = check_pinned_conditions(power_state, pinned=config.get("power.pinned"))
        profile_assertion = None

    result = measure_topology(config)
    passed = result.verdict == "pass"

    end_power_state = capture_power_state()

    # AF-006: a refused verification emits a manifest too. Its per-CPU scores are measurements of
    # real silicon and spec §9.2 requires them to be citable to a run_id; spec §5.1's treatment of
    # an UNSUPPORTED preflight is the same principle - a negative result is a data point.
    # A refused run records verified=false with null CPU lists, so the clustering it found can
    # never be read back as a mapping.
    run = emit(
        config=config,
        target="cpu-p",
        workload={
            "kind": "topology_verify",
            "benchmark": result.benchmark["kernel"],
            "task_ids": [],
            "seed": None,
            "n_repeats": result.benchmark["repeats_per_cpu"],
        },
        condition_label="topology_verify",
        allow_dirty=args.allow_dirty,
        repo_root=root,
        summary={
            **asdict(result),
            "session": {
                "power_state_start": asdict(power_state),
                "power_state_end": asdict(end_power_state),
                "pinned_condition_deviations": pinned_deviations,
            },
        },
        power_state=manifest_power_state(
            power_state,
            battery_pct_end=end_power_state.battery_pct,
            profile=profile_assertion,
        ),
        topology_override=(
            {"p_cpus": result.p_cpus, "lpe_cpus": result.lpe_cpus, "verified": True}
            if passed
            else {"p_cpus": None, "lpe_cpus": None, "verified": False}
        ),
        self_check="pass" if passed else "fail",
    )

    print(f"\nverdict            : {result.verdict.upper()}")
    print(f"fast cluster       : {result.p_cpus}")
    print(f"slow cluster       : {result.lpe_cpus}")
    print(f"separation ratio   : {result.cluster_separation_ratio:.3f}x")
    print(f"within-cluster CV  : fast {result.p_cluster_cv:.4f} / slow {result.lpe_cluster_cv:.4f}")
    print(f"scores (units/s)   : {result.scores}")
    print(f"EfficiencyClass map: {result.efficiency_class_map}")
    print(
        f"open question 4    : EfficiencyClass ordering "
        f"{'MATCHED' if result.efficiency_class_ordering_matched else 'DID NOT MATCH'} "
        f"the measured split (direction={result.efficiency_class_direction}, "
        f"partition_matched={result.efficiency_class_partition_matched})"
    )
    print(f"run_id             : {run.run_id}")

    if not passed:
        # The manifest exists, so the refusal is citable; the error still propagates, because a
        # refused verification must not look like a successful command.
        raise TopologyVerificationError(
            f"topology verification REFUSED (run_id={run.run_id}): "
            + " ".join(result.refusal_reasons)
            + " The mapping was NOT written to platform config."
        )

    if args.write:
        # A pass measured outside the pinned conditions is evidence, but it is not the value every
        # later run asserts. MACHINE.md calls such a session INVALID; this is where that bites.
        assert_pinned_for_committed_result(pinned_deviations)
        _write_verified_topology(
            root / "configs" / "platforms" / f"{args.platform_id}.yaml",
            result,
            run_id=run.run_id,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
