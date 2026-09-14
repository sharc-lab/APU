"""Confinement matrix: which mechanism actually pins inference to a core cluster?

The first affinity check found that ``SCHEDULING_CORE_TYPE=ECORE_ONLY`` confines work to the LP-E
cores while ``PCORE_ONLY`` silently spreads across all eight logical CPUs. This tool settles three
questions that finding raised:

1. Does ``PCORE_ONLY`` bind once ``ENABLE_CPU_PINNING`` is set **explicitly** (A2)? If so the
   original result is a *configuration* finding, not an OpenVINO defect, and the writeup changes.
2. Does process affinity confine both clusters (A5/A6)?
3. Does the confinement *mechanism* itself change throughput? A3 and A6 both run on CPUs 4-7 by
   different means, so a material tokens/s difference between them proves that mixing mechanisms
   across arms would confound silicon with confinement method.

**Every measurement runs in a fresh subprocess.** OpenVINO's TBB pool is built once per process and
process affinity is sticky, so running configurations in one interpreter would let each one
contaminate the next - and the contamination would look like a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seam.analysis.slice_stats import bootstrap_ci, coefficient_of_variation
from seam.backends.base import GenerationRequest, GenerationResult
from seam.backends.local_openvino import LocalOpenVinoBackend, runtime_info
from seam.config import load_platform_config
from seam.gitinfo import repo_root
from seam.model_provenance import load_local_spec, quantization_summary
from seam.tools.confinement_classify import (
    classify_cell,
    compute_noise_band,
    per_core_loaded_thresholds,
)
from seam.tools.verify_core_affinity import sample_per_cpu

__all__ = [
    "CONFIGS",
    "MatrixConfig",
    "assert_measurement_windows",
    "build_prompt",
    "check_forbidden_processes",
    "main",
    "schedule_blocks",
]

_RESULT_SENTINEL = "@@MATRIX_RESULT@@"
_BASELINE_S = 10.0
#: 100 Hz util sampling - CPU prefill TTFT on this IR is ~0.25 s (perf_metrics), so 10 Hz
#: cannot meet the >=10-sample prefill window gate. Decode remains long; overhead is recorded.
_SAMPLE_INTERVAL_S = 0.01
_FREQ_INTERVAL_S = 1.0
_SCORED_GENS = 2
_WARMUP_GENS = 1
_TOTAL_GENS = _WARMUP_GENS + _SCORED_GENS
_TARGET_PROMPT_TOKENS = 2048
_MAX_NEW_TOKENS = 128
#: Full matrix: fixed n=10 pre-registered. Verification uses ``--blocks 1``.
_DEFAULT_BLOCKS = 10
#: Minimum util samples required in each of prefill and decode for a scored generation.
_MIN_PHASE_SAMPLES = 10
#: Unconfined reference config ids - exempt from loaded-cores sanity; oversubscription pair.
_REFERENCE_CONFIGS = frozenset({"A0a", "A0b"})
#: Default inter-cell cooldown for the full 8x10 matrix (must be >60 s; thermal confound).
_DEFAULT_COOLDOWN_S = 120.0

_PARAGRAPH = (
    "Explain, in plain prose and without lists, how a deadline-aware scheduler decides whether "
    "to run a task locally or send it to a remote service. Cover latency prediction, the cost of "
    "being wrong in each direction, and what happens as the deadline tightens. "
)


@dataclass(frozen=True, slots=True)
class MatrixConfig:
    """One cell of the confinement matrix."""

    config_id: str
    cluster: str
    scheduling_core_type: str | None
    enable_cpu_pinning: bool | None
    affinity_cpus: tuple[int, ...] | None
    description: str
    inference_num_threads: int
    #: Unconfined references are exempt from the loaded-cores sanity gate.
    reference_exempt: bool = False


CONFIGS: dict[str, MatrixConfig] = {
    c.config_id: c
    for c in (
        MatrixConfig(
            "A0a",
            "all",
            None,
            None,
            None,
            "unconfined oversubscription reference: INFERENCE_NUM_THREADS=8 on default P-cores",
            inference_num_threads=8,
            reference_exempt=True,
        ),
        MatrixConfig(
            "A0b",
            "all",
            None,
            None,
            None,
            "unconfined default-placement reference: INFERENCE_NUM_THREADS=4",
            inference_num_threads=4,
            reference_exempt=True,
        ),
        MatrixConfig(
            "A1",
            "p",
            "PCORE_ONLY",
            None,
            None,
            "PCORE_ONLY, pinning default",
            inference_num_threads=4,
        ),
        MatrixConfig(
            "A2",
            "p",
            "PCORE_ONLY",
            True,
            None,
            "PCORE_ONLY, ENABLE_CPU_PINNING=YES",
            inference_num_threads=4,
        ),
        MatrixConfig(
            "A3",
            "lpe",
            "ECORE_ONLY",
            None,
            None,
            "ECORE_ONLY, pinning default",
            inference_num_threads=4,
        ),
        MatrixConfig(
            "A4",
            "lpe",
            "ECORE_ONLY",
            True,
            None,
            "ECORE_ONLY, ENABLE_CPU_PINNING=YES",
            inference_num_threads=4,
        ),
        MatrixConfig(
            "A5",
            "p",
            None,
            None,
            (0, 1, 2, 3),
            "process affinity 0-3, no core-type property",
            inference_num_threads=4,
        ),
        MatrixConfig(
            "A6",
            "lpe",
            None,
            None,
            (4, 5, 6, 7),
            "process affinity 4-7, no core-type property",
            inference_num_threads=4,
        ),
    )
}


def build_prompt(
    tokenizer: Any, *, target_tokens: int = _TARGET_PROMPT_TOKENS
) -> tuple[str, str, int]:
    """Build a deterministic long prompt and return ``(text, sha256, token_count)``."""
    text = _PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < target_tokens:
        text += _PARAGRAPH
    prompt_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    token_count = len(tokenizer(text)["input_ids"])
    return text, prompt_sha256, token_count


def _expected_cpus(cfg: MatrixConfig, platform_cfg: Any) -> list[int]:
    if cfg.cluster == "all":
        p = [int(c) for c in platform_cfg.get("topology.p_cpus") or ()]
        lpe = [int(c) for c in platform_cfg.get("topology.lpe_cpus") or ()]
        return sorted(p + lpe)
    if cfg.cluster == "p":
        return [int(c) for c in platform_cfg.get("topology.p_cpus") or ()]
    return [int(c) for c in platform_cfg.get("topology.lpe_cpus") or ()]


def _target_for(cfg: MatrixConfig) -> str:
    if cfg.cluster == "lpe":
        return "cpu-lpe"
    return "cpu-p"


def _mean_per_cpu(samples: Sequence[Sequence[float]]) -> list[float]:
    if not samples:
        return []
    n_cpus = len(samples[0])
    return [statistics.fmean(row[cpu] for row in samples) for cpu in range(n_cpus)]


def _sample_baseline(*, duration_s: float, interval_s: float) -> list[list[float]]:
    stop = threading.Event()
    holder: list[list[float]] = []

    def _run() -> None:
        holder.extend(sample_per_cpu(stop, interval_s))

    thread = threading.Thread(target=_run, name="idle-baseline", daemon=True)
    thread.start()
    time.sleep(duration_s)
    stop.set()
    thread.join(timeout=5.0)
    return holder


@dataclass(slots=True)
class _TimedSample:
    t_ns: int
    pct: list[float]


def _generate_with_utilization(
    backend: LocalOpenVinoBackend,
    request: GenerationRequest,
    *,
    n_cpus: int = 8,
) -> tuple[GenerationResult, list[_TimedSample], dict[str, Any], int]:
    """Run one generation while sampling per-CPU utilization (10 Hz) and frequency (>=1 Hz).

    The sampler runs **continuously** across the entire ``generate()`` call. Prefill/decode
    boundaries are applied **post hoc** from the first-token timestamp. Never gate the sampler
    on an event that ends a phase (that is what emptied all 70 prefill windows when TTFT was
    unavailable - every sample fell into decode).

    Returns ``(result, samples, overhead, ttft_split_ns)`` where ``ttft_split_ns`` is the
    first-token boundary on the util-sampler clock (generate-start offset + backend TTFT).
    """
    from seam.telemetry.frequency import FrequencySampler

    samples: list[_TimedSample] = []
    stop = threading.Event()
    t0 = time.perf_counter_ns()
    sampler_wakeups = 0
    generate_start_offset_ns = 0
    generate_started = threading.Event()

    def _sampler() -> None:
        nonlocal sampler_wakeups
        import psutil

        psutil.cpu_percent(percpu=True)  # prime; discarded
        # Wait until generate() has started so we do not dilute prefill with idle samples,
        # then sample continuously (sample-then-sleep) until stop.
        generate_started.wait(timeout=600.0)
        while not stop.is_set():
            sampler_wakeups += 1
            samples.append(
                _TimedSample(
                    time.perf_counter_ns() - t0,
                    list(psutil.cpu_percent(percpu=True)),
                )
            )
            # Interruptible sleep - never gate on first-token / phase-end events.
            if stop.wait(_SAMPLE_INTERVAL_S):
                break

    freq = FrequencySampler(interval_s=_FREQ_INTERVAL_S, n_cpus=n_cpus)
    thread = threading.Thread(target=_sampler, name="gen-util", daemon=True)
    thread.start()
    freq.start()
    try:
        generate_start_offset_ns = time.perf_counter_ns() - t0
        generate_started.set()
        result = backend.generate(request, ignore_eos=True)
    finally:
        stop.set()
        generate_started.set()
        thread.join(timeout=5.0)
        freq_samples = freq.stop()
    wall_s = result.wall_ns / 1e9
    expected_wakeups = wall_s / _SAMPLE_INTERVAL_S if wall_s > 0 else 0.0
    ttft_ns = result.ttft_ns or 0
    ttft_split_ns = generate_start_offset_ns + ttft_ns if ttft_ns > 0 else 0
    overhead = {
        "util_sampler_method": "psutil.cpu_percent(percpu=True)",
        "util_sample_interval_s": _SAMPLE_INTERVAL_S,
        "util_sampler_mode": "continuous_across_generate_posthoc_ttft_split",
        "util_sampler_wakeups": sampler_wakeups,
        "util_sampler_expected_wakeups": round(expected_wakeups, 2),
        "util_sampler_overhead_wakeup_ratio": (
            round(sampler_wakeups / expected_wakeups, 4) if expected_wakeups > 0 else None
        ),
        "generate_start_offset_ns": generate_start_offset_ns,
        "ttft_split_ns": ttft_split_ns,
        "frequency": {
            "interval_s": _FREQ_INTERVAL_S,
            "n_samples": len(freq_samples),
            "summary": freq.summary(),
            "method": freq_samples[0].method if freq_samples else "none",
        },
    }
    return result, samples, overhead, ttft_split_ns


def assert_measurement_windows(
    *,
    n_prefill: int,
    n_decode: int,
    ttft_s: float,
    decode_s: float,
    min_samples: int = 1,
) -> None:
    """Hard-fail a cell when any declared measurement window is empty or zero-duration.

    Empty/zero-duration windows must never become nulls inside a completed ``ok=True`` record.
    """
    if n_prefill < min_samples:
        raise RuntimeError(
            f"HARD FAILURE: prefill util window empty/short "
            f"(n_prefill={n_prefill}, min={min_samples})"
        )
    if n_decode < min_samples:
        raise RuntimeError(
            f"HARD FAILURE: decode util window empty/short (n_decode={n_decode}, min={min_samples})"
        )
    if ttft_s <= 0.0:
        raise RuntimeError(f"HARD FAILURE: prefill duration non-positive (ttft_s={ttft_s})")
    if decode_s <= 0.0:
        raise RuntimeError(f"HARD FAILURE: decode duration non-positive (decode_s={decode_s})")


def _phase_means(
    samples: Sequence[_TimedSample],
    ttft_ns: int | None,
) -> tuple[list[float], list[float], list[float], int, int]:
    """Return ``(overall, prefill, decode, n_prefill, n_decode)`` per-core means.

    Prefill and decode are never pooled. Missing/zero TTFT yields empty phase means (not
    "everything is decode") so classification cannot silently adopt from a collapsed boundary.
    """
    if not samples:
        return [], [], [], 0, 0
    all_rows = [s.pct for s in samples]
    overall = _mean_per_cpu(all_rows)
    if ttft_ns is None or ttft_ns <= 0:
        return overall, [], [], 0, 0
    prefill_rows = [s.pct for s in samples if s.t_ns < ttft_ns]
    decode_rows = [s.pct for s in samples if s.t_ns >= ttft_ns]
    return (
        overall,
        _mean_per_cpu(prefill_rows),
        _mean_per_cpu(decode_rows),
        len(prefill_rows),
        len(decode_rows),
    )


def _deltas(phase_mean: list[float], baseline_mean: list[float]) -> list[float]:
    if not phase_mean:
        # Empty phase → empty deltas. Padding zeros against a non-zero baseline invents
        # large negative "prefill" load and forces INVALID for the wrong reason.
        return []
    n = max(len(phase_mean), len(baseline_mean))
    phase = phase_mean + [0.0] * (n - len(phase_mean))
    base = baseline_mean + [0.0] * (n - len(baseline_mean))
    return [phase[cpu] - base[cpu] for cpu in range(n)]


def _generation_record(
    *,
    gen_index: int,
    baseline_samples: list[list[float]],
    result: GenerationResult,
    util_samples: list[_TimedSample],
    sampler_overhead: dict[str, Any],
    ttft_split_ns: int | None = None,
    min_phase_samples: int = 1,
) -> dict[str, Any]:
    baseline_mean = _mean_per_cpu(baseline_samples)
    ttft_ns = result.ttft_ns
    if ttft_ns is None or ttft_ns <= 0:
        raise RuntimeError(
            "generation returned missing/zero TTFT; prefill/decode split would be fabricated"
        )
    # Prefer sampler-clock boundary (generate start offset + backend TTFT) when provided.
    split_ns = ttft_split_ns if ttft_split_ns is not None and ttft_split_ns > 0 else ttft_ns
    _overall, prefill, decode, n_prefill, n_decode = _phase_means(util_samples, split_ns)
    ttft_s = ttft_ns / 1e9
    wall_s = result.wall_ns / 1e9
    decode_s = wall_s - ttft_s
    # Fail loud BEFORE assembling a completed record - empty windows are not nulls.
    assert_measurement_windows(
        n_prefill=n_prefill,
        n_decode=n_decode,
        ttft_s=ttft_s,
        decode_s=decode_s,
        min_samples=min_phase_samples,
    )

    decode_delta = _deltas(decode, baseline_mean)
    prefill_delta = _deltas(prefill, baseline_mean)

    return {
        "gen_index": gen_index,
        "scored": True,
        "ok": True,
        "n_baseline_samples": len(baseline_samples),
        "n_util_samples": len(util_samples),
        "n_prefill_util_samples": n_prefill,
        "n_decode_util_samples": n_decode,
        "prefill_duration_s": ttft_s,
        "decode_duration_s": decode_s,
        "baseline_mean_pct_per_cpu": [round(v, 2) for v in baseline_mean],
        "prefill_mean_pct_per_cpu": [round(v, 2) for v in prefill],
        "decode_mean_pct_per_cpu": [round(v, 2) for v in decode],
        "prefill_delta_pct_per_cpu": [round(v, 2) for v in prefill_delta],
        "decode_delta_pct_per_cpu": [round(v, 2) for v in decode_delta],
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "wall_s": wall_s,
        "ttft_s": ttft_s,
        "ttft_ns": ttft_ns,
        "ttft_split_ns": split_ns,
        "ttft_source": (result.extra or {}).get("ttft_source"),
        "ttft_ns_perf_metrics": (result.extra or {}).get("ttft_ns_perf_metrics"),
        "ttft_ns_streamer": (result.extra or {}).get("ttft_ns_streamer"),
        "r_prefill_tok_s": result.prompt_tokens / ttft_s,
        "r_decode_tok_s": result.completion_tokens / decode_s,
        "sampler_overhead": sampler_overhead,
    }


def run_one(
    config_id: str, *, spec_path: Path, platform: str, threads: int | None = None
) -> dict[str, Any]:
    """Execute a single matrix cell. Called in the CHILD process."""
    cfg = CONFIGS[config_id]
    thread_count = int(threads) if threads is not None else cfg.inference_num_threads
    root = repo_root(Path(__file__).parent)
    spec = load_local_spec(spec_path)
    platform_cfg = load_platform_config(platform, repo_root=root)
    expected = _expected_cpus(cfg, platform_cfg)
    target = _target_for(cfg)

    backend = LocalOpenVinoBackend(
        model_dir=Path(spec["ir_dir"]),
        target=target,  # type: ignore[arg-type]
        scheduling_core_type=cfg.scheduling_core_type,
        inference_num_threads=thread_count,
        enable_cpu_pinning=cfg.enable_cpu_pinning,
        model_ref=f"{spec['name']}@{str(spec['revision'])[:12]}+{quantization_summary(spec)}",
        enable_thinking=False,
        affinity_cpus=list(cfg.affinity_cpus) if cfg.affinity_cpus else None,
    )
    verdict = backend.preflight()
    if verdict.status != "OK":
        return {
            "config_id": config_id,
            "ok": False,
            "preflight": verdict.status,
            "reason": verdict.reason,
        }

    backend.load()
    prompt_text, prompt_sha256, prompt_tokens = build_prompt(backend.tokenizer)
    request = GenerationRequest(
        messages=[{"role": "user", "content": prompt_text}],
        system="You are a helpful assistant.",
        tools=(),
        max_tokens=_MAX_NEW_TOKENS,
    )

    import psutil

    try:
        affinity_mask = sorted(psutil.Process().cpu_affinity())
    except Exception:
        affinity_mask = []

    generations: list[dict[str, Any]] = []
    for gen_index in range(_TOTAL_GENS):
        baseline_samples: list[list[float]] = []
        if gen_index >= _WARMUP_GENS:
            baseline_samples = _sample_baseline(
                duration_s=_BASELINE_S, interval_s=_SAMPLE_INTERVAL_S
            )
        result, util_samples, overhead, ttft_split_ns = _generate_with_utilization(
            backend, request, n_cpus=8
        )
        if gen_index >= _WARMUP_GENS:
            # Scored gens require non-empty windows; raise → child non-zero → cell not ok.
            generations.append(
                _generation_record(
                    gen_index=gen_index,
                    baseline_samples=baseline_samples,
                    result=result,
                    util_samples=util_samples,
                    sampler_overhead=overhead,
                    ttft_split_ns=ttft_split_ns,
                    min_phase_samples=1,
                )
            )

    return {
        "config_id": config_id,
        "ok": True,
        "description": cfg.description,
        "target": target,
        "expected_cpus": expected,
        "inference_num_threads": thread_count,
        "preflight": "OK",
        "config": backend.config_record(),
        "prompt_sha256": prompt_sha256,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": _MAX_NEW_TOKENS,
        "process_affinity_mask": affinity_mask,
        "generations": generations,
        "runtime": {"openvino": runtime_info().openvino, "genai": runtime_info().genai},
    }


def check_forbidden_processes(patterns: Iterable[str]) -> list[dict[str, str]]:
    """Return processes whose cmdline indicates a forbidden *running tool*, not a string mention.

    Substring match alone is wrong: a quiesce checker that *lists* ``fetch_progress`` in its
    own argv would refuse itself. Match module invocation (``-m seam.tools.fetch_progress``)
    or a path segment that is the script being executed.
    """
    import psutil

    hits: list[dict[str, str]] = []
    self_pid = psutil.Process().pid
    for proc in psutil.process_iter(["pid", "cmdline"]):
        pid = proc.info.get("pid")
        if pid == self_pid:
            continue
        cmdline_list = [str(part) for part in (proc.info.get("cmdline") or [])]
        if not cmdline_list:
            continue
        cmdline = " ".join(cmdline_list)
        # Skip processes that are only *inspecting* forbidden names (this checker, audits).
        if "check_forbidden_processes" in cmdline:
            continue
        for pattern in patterns:
            module_form = f"-m {pattern}" if not pattern.endswith(".exe") else None
            script_hit = any(
                part.endswith(pattern) or part.endswith(pattern.replace(".", "/") + ".py")
                for part in cmdline_list
            )
            module_hit = bool(module_form and module_form in cmdline)
            exe_hit = pattern.lower().endswith(".exe") and any(
                part.lower().endswith(pattern.lower()) for part in cmdline_list
            )
            if module_hit or script_hit or exe_hit:
                hits.append({"pid": str(pid), "pattern": pattern, "cmdline": cmdline})
                break
    return hits


def schedule_blocks(
    config_ids: Sequence[str],
    *,
    n_blocks: int,
    seed: int,
) -> tuple[list[list[str]], int]:
    """Build ``n_blocks`` shuffled cell orders with no consecutive duplicate across blocks."""
    rng = random.Random(seed)
    blocks: list[list[str]] = []
    prev_last: str | None = None
    for _ in range(n_blocks):
        order = _shuffle_no_consecutive(list(config_ids), rng, avoid_first=prev_last)
        blocks.append(order)
        prev_last = order[-1]
    return blocks, seed


def _shuffle_no_consecutive(
    items: list[str],
    rng: random.Random,
    *,
    avoid_first: str | None,
) -> list[str]:
    for _ in range(10_000):
        order = items[:]
        rng.shuffle(order)
        if avoid_first is not None and order and order[0] == avoid_first:
            continue
        if all(order[i] != order[i + 1] for i in range(len(order) - 1)):
            return order
    raise RuntimeError("could not shuffle without consecutive duplicates")


def _spawn(
    config_id: str, *, spec_path: Path, platform: str, threads: int | None = None
) -> dict[str, Any]:
    cfg = CONFIGS[config_id]
    thread_count = int(threads) if threads is not None else cfg.inference_num_threads
    command = [
        sys.executable,
        "-m",
        "seam.tools.affinity_matrix",
        "--child",
        "--config-id",
        config_id,
        "--spec",
        str(spec_path),
        "--platform",
        platform,
        "--threads",
        str(thread_count),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        # 4294967295 == unsigned -1: typically killed (OOM, sleep, or AC-loss shutdown).
        return {
            "config_id": config_id,
            "error": f"child exited {completed.returncode}",
            "returncode": completed.returncode,
            "stderr": (completed.stderr or "")[-4000:],
            "stdout_tail": (completed.stdout or "")[-2000:],
        }
    for line in completed.stdout.splitlines():
        if line.startswith(_RESULT_SENTINEL):
            parsed: dict[str, Any] = json.loads(line[len(_RESULT_SENTINEL) :])
            return parsed
    return {
        "config_id": config_id,
        "error": "child produced no result line",
        "stdout": (completed.stdout or "")[-2000:],
        "stderr": (completed.stderr or "")[-2000:],
    }


def _write_checkpoint(path: Path, report: dict[str, Any]) -> None:
    """Durable per-cell checkpoint so a killed parent does not erase the whole matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = dict(report)
    payload["checkpoint"] = True
    data = json.dumps(payload, indent=2)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        import os

        os.fsync(fh.fileno())
    tmp.replace(path)


def _completed_cells(runs: dict[str, list[dict[str, Any]]]) -> set[tuple[int, str]]:
    """Return ``{(block_index, config_id)}`` for cells with ok scored generations."""
    done: set[tuple[int, str]] = set()
    for config_id, cell_runs in runs.items():
        for run in cell_runs:
            if _run_is_ok(run):
                block = run.get("block")
                if isinstance(block, int):
                    done.add((block, config_id))
    return done


def _keep_awake_windows() -> Any:
    """Prevent AC sleep while the matrix runs (Windows SetThreadExecutionState)."""
    if sys.platform != "win32":
        return None
    import ctypes

    es_continuous = 0x80000000
    es_system_required = 0x00000001
    es_awaymode_required = 0x00000040
    flags = es_continuous | es_system_required | es_awaymode_required
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    return flags


def _release_awake_windows(flags: Any) -> None:
    if flags is None or sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def _aggregate_deltas(cell_runs: list[dict[str, Any]], phase_key: str) -> dict[int, float]:
    """Mean per-core delta for one phase across scored generations in successful runs."""
    per_cpu: dict[int, list[float]] = {}
    for run in cell_runs:
        for gen in run.get("generations") or []:
            for cpu, delta in enumerate(gen.get(phase_key) or []):
                per_cpu.setdefault(cpu, []).append(float(delta))
    return {cpu: statistics.fmean(vals) for cpu, vals in per_cpu.items()}


def _pool_baselines(all_runs: dict[str, list[dict[str, Any]]]) -> dict[int, list[float]]:
    pooled: dict[int, list[float]] = {}
    for runs in all_runs.values():
        for run in runs:
            for gen in run.get("generations") or []:
                baseline = gen.get("baseline_mean_pct_per_cpu") or []
                for cpu, value in enumerate(baseline):
                    pooled.setdefault(cpu, []).append(float(value))
    return pooled


def _run_is_ok(run: dict[str, Any]) -> bool:
    """A cell run is ok only when every scored gen has non-empty positive-duration windows."""
    if run.get("ok") is False or run.get("error"):
        return False
    gens = run.get("generations") or []
    if not gens:
        return False
    for gen in gens:
        if not gen.get("ok", False):
            return False
        if int(gen.get("n_prefill_util_samples") or 0) < 1:
            return False
        if int(gen.get("n_decode_util_samples") or 0) < 1:
            return False
        if float(gen.get("prefill_duration_s") or 0.0) <= 0.0:
            return False
        if float(gen.get("decode_duration_s") or 0.0) <= 0.0:
            return False
        if not gen.get("prefill_delta_pct_per_cpu"):
            return False
        if not gen.get("decode_delta_pct_per_cpu"):
            return False
    return True


def _summarize_cell(
    config_id: str,
    cell_runs: list[dict[str, Any]],
    *,
    noise_band: float,
    all_cpus: set[int],
    platform_cfg: Any,
    loaded_thresholds: dict[int, float],
) -> dict[str, Any]:
    ok = [r for r in cell_runs if _run_is_ok(r)]
    cfg = CONFIGS[config_id]
    requested = set(_expected_cpus(cfg, platform_cfg))
    is_reference = config_id in _REFERENCE_CONFIGS or cfg.reference_exempt

    # Prefill and decode are classified SEPARATELY - confinement may hold in one phase only.
    # Never pool the two phases for a single verdict.
    prefill_deltas = _aggregate_deltas(ok, "prefill_delta_pct_per_cpu")
    decode_deltas = _aggregate_deltas(ok, "decode_delta_pct_per_cpu")
    prefill_empty = not prefill_deltas
    decode_empty = not decode_deltas
    thr_report = {str(k): round(v, 4) for k, v in sorted(loaded_thresholds.items())}

    def _empty_phase(reason: str) -> dict[str, Any]:
        return {
            "requested_cpus": sorted(requested),
            "loaded_threshold": thr_report,
            "noise_band": noise_band,
            "core_states": {},
            "verdict": "N/A" if is_reference else "INVALID",
            "reason": reason,
        }

    if prefill_empty:
        prefill_cls = _empty_phase("empty_prefill_util_samples")
    else:
        prefill_cls = classify_cell(
            prefill_deltas,
            requested,
            noise_band=noise_band,
            all_cpus=all_cpus,
            loaded_thresholds=loaded_thresholds,
        )
    if decode_empty:
        decode_cls = _empty_phase("empty_decode_util_samples")
    else:
        decode_cls = classify_cell(
            decode_deltas,
            requested,
            noise_band=noise_band,
            all_cpus=all_cpus,
            loaded_thresholds=loaded_thresholds,
        )

    if is_reference:
        overall_verdict = "N/A"
    elif prefill_cls["verdict"] == "INVALID" or decode_cls["verdict"] == "INVALID":
        overall_verdict = "INVALID"
    elif prefill_cls["verdict"] == "LEAKED" or decode_cls["verdict"] == "LEAKED":
        overall_verdict = "LEAKED"
    elif prefill_cls["verdict"] == "UNCLEAR" or decode_cls["verdict"] == "UNCLEAR":
        overall_verdict = "UNCLEAR"
    elif prefill_cls["verdict"] == "CONFINED" and decode_cls["verdict"] == "CONFINED":
        overall_verdict = "CONFINED"
    else:
        overall_verdict = "UNCLEAR"

    prefill_rates: list[float] = []
    decode_rates: list[float] = []
    for run in ok:
        for gen in run["generations"]:
            if gen.get("r_prefill_tok_s") is not None:
                prefill_rates.append(float(gen["r_prefill_tok_s"]))
            if gen.get("r_decode_tok_s") is not None:
                decode_rates.append(float(gen["r_decode_tok_s"]))

    block_prefill: list[float] = []
    block_decode: list[float] = []
    for run in ok:
        gens = run["generations"]
        if gens:
            prefill_vals = [float(g["r_prefill_tok_s"]) for g in gens if g.get("r_prefill_tok_s")]
            decode_vals = [float(g["r_decode_tok_s"]) for g in gens if g.get("r_decode_tok_s")]
            if prefill_vals:
                block_prefill.append(statistics.fmean(prefill_vals))
            if decode_vals:
                block_decode.append(statistics.fmean(decode_vals))

    summary: dict[str, Any] = {
        "n_runs": len(cell_runs),
        "n_ok": len(ok),
        "inference_num_threads": cfg.inference_num_threads,
        "classification_prefill": prefill_cls,
        "classification_decode": decode_cls,
        "classification": {
            "verdict": overall_verdict,
            "prefill": prefill_cls["verdict"],
            "decode": decode_cls["verdict"],
        },
        "mean_prefill_delta_pct_per_cpu": [
            round(prefill_deltas.get(cpu, 0.0), 2) for cpu in sorted(all_cpus)
        ],
        "mean_decode_delta_pct_per_cpu": [
            round(decode_deltas.get(cpu, 0.0), 2) for cpu in sorted(all_cpus)
        ],
        "r_prefill_tok_s_all": prefill_rates,
        "r_decode_tok_s_all": decode_rates,
        "r_prefill_tok_s_mean": statistics.fmean(prefill_rates) if prefill_rates else None,
        "r_decode_tok_s_mean": statistics.fmean(decode_rates) if decode_rates else None,
        "r_prefill_tok_s_ci": bootstrap_ci(block_prefill, label="prefill_tok_s").to_dict()
        if len(block_prefill) >= 2
        else None,
        "r_decode_tok_s_ci": bootstrap_ci(block_decode, label="decode_tok_s").to_dict()
        if len(block_decode) >= 2
        else None,
    }

    if is_reference:
        # A0a/A0b are EXEMPT from loaded-cores sanity. Record placement diagnostics only.
        # Purpose: A0a vs A0b oversubscription cost (threads=8 vs 4 on the same four P-cores
        # OpenVINO chooses by default). Do NOT gate on A0a saturating all 8 cores.
        core_states = decode_cls.get("core_states", {})
        assert isinstance(core_states, dict)
        loaded_cpus = sorted(int(cpu) for cpu, state in core_states.items() if state == "LOADED")
        summary["reference_diagnostics"] = {
            "config_id": config_id,
            "threads": cfg.inference_num_threads,
            "n_logical_cpus": len(all_cpus),
            "loaded_core_count": len(loaded_cpus),
            "loaded_cpus": loaded_cpus,
            "exempt_from_loaded_cores_sanity": True,
            "note": (
                "A0a (threads=8) vs A0b (threads=4) measures oversubscription cost on the "
                "P-cores OpenVINO selects by default. Neither gates adoption via loaded-core "
                "count. Per-core LOADED thresholds come from A5/A6 decode deltas (two-pass)."
            ),
        }

    return summary


def _mechanism_decision(summary: dict[str, Any]) -> dict[str, Any]:
    """Pick the ONE mechanism that confines both clusters in BOTH prefill and decode."""

    def _confined_both_phases(cid: str) -> bool:
        cls = summary.get(cid, {}).get("classification", {})
        return bool(cls.get("prefill") == "CONFINED" and cls.get("decode") == "CONFINED")

    def _phase_pair(cid: str) -> dict[str, str]:
        cls = summary.get(cid, {}).get("classification", {})
        return {
            "prefill": str(cls.get("prefill", "missing")),
            "decode": str(cls.get("decode", "missing")),
            "overall": str(cls.get("verdict", "missing")),
        }

    a1 = _phase_pair("A1")
    a2 = _phase_pair("A2")
    a3 = _phase_pair("A3")
    a4 = _phase_pair("A4")
    a5_confined = _confined_both_phases("A5")
    a6_confined = _confined_both_phases("A6")
    a1_confined = _confined_both_phases("A1")
    a2_confined = _confined_both_phases("A2")
    a3_confined = _confined_both_phases("A3")
    a4_confined = _confined_both_phases("A4")

    # Decision tree: identify mechanisms where BOTH {0-3} and {4-7} are CONFINED in both phases.
    candidates: list[tuple[str, str]] = []
    # Native: prefer the pinned variants (A2/A4) when both confined; else any confined pair.
    if a2_confined and a4_confined:
        candidates.append(
            (
                "scheduling-core-type",
                "A2+A4 CONFINED in prefill and decode (ENABLE_CPU_PINNING=YES).",
            )
        )
    elif a1_confined and a3_confined:
        candidates.append(
            (
                "scheduling-core-type",
                "A1+A3 CONFINED in prefill and decode (pinning default).",
            )
        )
    if a5_confined and a6_confined:
        candidates.append(
            (
                "process-affinity",
                "A5+A6 CONFINED in prefill and decode.",
            )
        )

    if not candidates:
        chosen, reason = (
            "none",
            "Zero mechanisms confined BOTH clusters in BOTH prefill and decode. STOP.",
        )
    elif len(candidates) == 1:
        chosen, reason = candidates[0]
    else:
        # More than one: adopt higher measured throughput (lower confinement overhead).
        native_decode = statistics.fmean(
            [
                v
                for v in (
                    summary.get("A2", {}).get("r_decode_tok_s_mean"),
                    summary.get("A4", {}).get("r_decode_tok_s_mean"),
                )
                if isinstance(v, (int, float))
            ]
            or [0.0]
        )
        affinity_decode = statistics.fmean(
            [
                v
                for v in (
                    summary.get("A5", {}).get("r_decode_tok_s_mean"),
                    summary.get("A6", {}).get("r_decode_tok_s_mean"),
                )
                if isinstance(v, (int, float))
            ]
            or [0.0]
        )
        if affinity_decode > native_decode:
            chosen, reason = (
                "process-affinity",
                f"Both mechanisms confine; process-affinity higher decode tok/s "
                f"({affinity_decode:.3f} > {native_decode:.3f}).",
            )
        else:
            chosen, reason = (
                "scheduling-core-type",
                f"Both mechanisms confine; scheduling-core-type higher decode tok/s "
                f"({native_decode:.3f} >= {affinity_decode:.3f}).",
            )

    # A2 classification: defect vs configuration vs non-reproduction.
    if a1["overall"] == "CONFINED":
        a2_label = "non_reproduction"
        a2_note = (
            "A1 CONFINED under baseline subtraction; original PCORE_ONLY fall-through finding "
            "does not reproduce. Retract in AUDIT_LOG."
        )
    elif a1["overall"] == "LEAKED" and a2["overall"] == "CONFINED":
        a2_label = "configuration"
        a2_note = (
            "A1 LEAKED and A2 CONFINED: PCORE_ONLY is advisory unless ENABLE_CPU_PINNING is set."
        )
    elif a1["overall"] == "LEAKED" and a2["overall"] == "LEAKED":
        a2_label = "defect"
        a2_note = "A1 LEAKED and A2 LEAKED: PCORE_ONLY does not bind on Panther Lake."
    else:
        a2_label = "inconclusive"
        a2_note = f"A1={a1}; A2={a2}. Increase replicates if UNCLEAR."

    a3_mean = summary.get("A3", {}).get("r_decode_tok_s_mean")
    a6_mean = summary.get("A6", {}).get("r_decode_tok_s_mean")
    a3_prefill = summary.get("A3", {}).get("r_prefill_tok_s_mean")
    a6_prefill = summary.get("A6", {}).get("r_prefill_tok_s_mean")
    a3_all: list[float] = summary.get("A3", {}).get("r_decode_tok_s_all") or []
    a6_all: list[float] = summary.get("A6", {}).get("r_decode_tok_s_all") or []
    cv_a3 = coefficient_of_variation(a3_all)
    cv_a6 = coefficient_of_variation(a6_all)
    within_cv = max(v for v in (cv_a3, cv_a6) if v == v) if a3_all and a6_all else float("nan")

    mechanism_effect: dict[str, Any] = {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "a2_label": a2_label,
        "a2_note": a2_note,
        "a3_ecore_only_decode_tok_s": a3_mean,
        "a6_process_affinity_decode_tok_s": a6_mean,
        "a3_ecore_only_prefill_tok_s": a3_prefill,
        "a6_process_affinity_prefill_tok_s": a6_prefill,
        "a3_within_cv": None if cv_a3 != cv_a3 else round(cv_a3, 4),
        "a6_within_cv": None if cv_a6 != cv_a6 else round(cv_a6, 4),
        "candidates": [{"mechanism": m, "reason": r} for m, r in candidates],
    }
    if a3_mean and a6_mean and within_cv == within_cv:
        ratio = a3_mean / a6_mean
        rel_delta = abs(ratio - 1.0)
        threshold = 2.0 * within_cv
        material = rel_delta > threshold
        mechanism_effect["ratio_a3_over_a6_decode"] = round(ratio, 4)
        mechanism_effect["relative_delta_decode"] = round(rel_delta, 4)
        mechanism_effect["material_threshold_2x_cv"] = round(threshold, 4)
        mechanism_effect["material"] = material
        mechanism_effect["interpretation"] = (
            "Mechanism affects throughput on identical cores; symmetry is proven NECESSARY."
            if material
            else "No material throughput difference on identical cores at >2x within-cell CV; "
            "symmetry remains a design requirement on isolation-invariant grounds."
        )
    return {
        "adopted_mechanism": chosen,
        "reason": reason,
        "mechanism_effect": mechanism_effect,
        "a2_verdict": {"label": a2_label, "note": a2_note, "a1": a1, "a2": a2},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--platform", default="aipc-c1")
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Override INFERENCE_NUM_THREADS for a child cell. Parent uses each "
            "MatrixConfig.inference_num_threads (A0a=8, A0b=4, A1-A6=4)."
        ),
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=_DEFAULT_BLOCKS,
        help=f"replicate blocks (default {_DEFAULT_BLOCKS}; use 1 for verification)",
    )
    parser.add_argument("--seed", type=int, default=20260803, help="shuffle seed (recorded)")
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=_DEFAULT_COOLDOWN_S,
        help=(
            f"seconds between cells (default {_DEFAULT_COOLDOWN_S:.0f}; "
            "must be >60 for full matrix thermal)"
        ),
    )
    parser.add_argument(
        "--allow-battery",
        action="store_true",
        help="OVERRIDE: run on battery (violates quiesce; do not use for publishable runs)",
    )
    parser.add_argument(
        "--allow-charging",
        action="store_true",
        help="OVERRIDE: allow charging!=false (violates ac-pinned charging-complete)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from out-stem.partial.json, skipping cells that already have generations",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        default=True,
        help="Permit dirty git tree for sealed manifest emit (recorded in manifest)",
    )
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    spec_path = args.spec or (root / "configs" / "models" / "Qwen3-4B-int4-ov.yaml")

    if args.child:
        assert args.config_id is not None
        result = run_one(
            args.config_id, spec_path=spec_path, platform=args.platform, threads=args.threads
        )
        print(_RESULT_SENTINEL + json.dumps(result))
        return 0

    if args.blocks >= _DEFAULT_BLOCKS and float(args.cooldown_s) <= 60.0:
        print(
            f"REFUSED: full matrix (blocks={args.blocks}) requires cooldown_s > 60 "
            f"(got {args.cooldown_s}). Thermal confound over 80 cells. STOP."
        )
        return 2

    from seam.powerstate import (
        assert_profile,
        capture_battery_status_wmi,
        capture_power_state,
        is_charging_complete,
        raise_if_profile_mismatch,
    )

    forbidden_patterns = (
        "seam.tools.fetch_progress",
        "fetch_progress",
        "seam.tools.phase_e_cloud",
        "curl.exe",
        "wget",
        "TiWorker.exe",
        "UsoClient.exe",
        "OneDrive.exe",
        "MsMpEng.exe",
    )
    hard_refuse_patterns = (
        "seam.tools.fetch_progress",
        "fetch_progress",
        "seam.tools.phase_e_cloud",
        "curl.exe",
    )
    forbidden_hits = check_forbidden_processes(forbidden_patterns)
    hard_hits = [h for h in forbidden_hits if h["pattern"] in hard_refuse_patterns]
    if hard_hits:
        print("REFUSED: forbidden processes running:", json.dumps(hard_hits, indent=2))
        return 2

    platform_cfg = load_platform_config(args.platform, repo_root=root)
    power_cfg = platform_cfg.get("power") or {}
    ac_profile = (power_cfg.get("profiles") or {}).get("ac-pinned") or {}
    brightness_target = ac_profile.get("display_brightness_pct")
    charge_rate_max = ac_profile.get("charge_rate_max_mw")
    soc_complete = ac_profile.get("charging_complete_soc_pct")

    power = capture_power_state()
    batt_wmi = capture_battery_status_wmi()
    if power.on_battery and not args.allow_battery:
        print(
            "REFUSED: AC not connected (on_battery=True, "
            f"battery_pct={power.battery_pct}, charging={power.charging}). "
            "Quiesce requires AC. Plug in and re-run. This is a gate, not a suggestion."
        )
        refuse_path = root / "derived" / "mslice" / "affinity_matrix_refused_ac.json"
        refuse_path.parent.mkdir(parents=True, exist_ok=True)
        refuse_path.write_text(
            json.dumps(
                {
                    "refused": True,
                    "reason": "ac_not_connected",
                    "quiesce": {
                        "on_battery": power.on_battery,
                        "battery_pct": power.battery_pct,
                        "charging": power.charging,
                        "power_plan_name": power.power_plan_name,
                        "power_plan_guid": power.power_plan_guid,
                        "battery_status_wmi": batt_wmi.__dict__,
                    },
                    "thermal": {"regime": "confound"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {refuse_path}")
        return 3

    complete, complete_reason = is_charging_complete(
        power,
        batt_wmi,
        charge_rate_max_mw=float(charge_rate_max) if charge_rate_max is not None else None,
        charging_complete_soc_pct=float(soc_complete) if soc_complete is not None else None,
    )
    if not complete and not args.allow_charging:
        print(
            "REFUSED: ac-pinned requires charging complete. "
            f"reason={complete_reason}. Do NOT measure while charging. STOP."
        )
        refuse_path = root / "derived" / "mslice" / "affinity_matrix_refused_charging.json"
        refuse_path.parent.mkdir(parents=True, exist_ok=True)
        refuse_path.write_text(
            json.dumps(
                {
                    "refused": True,
                    "reason": "charging_not_complete",
                    "detail": complete_reason,
                    "quiesce": {
                        "on_battery": power.on_battery,
                        "battery_pct": power.battery_pct,
                        "charging": power.charging,
                        "battery_status_wmi": batt_wmi.__dict__,
                    },
                    "thermal": {"regime": "confound"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {refuse_path}")
        return 5

    profile_assertion = assert_profile("affinity_matrix", power, power_cfg=power_cfg)
    if profile_assertion.deviations and not args.allow_charging:
        print("REFUSED: ac-pinned profile mismatch:", profile_assertion.deviations)
        try:
            raise_if_profile_mismatch(profile_assertion)
        except Exception as exc:
            print(str(exc))
        return 5

    brightness_set = _set_display_brightness(
        int(brightness_target) if brightness_target is not None else None
    )
    quiesce_extra = _capture_quiesce_extras()
    quiesce_extra["display_brightness_target"] = brightness_target
    quiesce_extra["display_brightness_set"] = brightness_set
    quiesce_extra["display_brightness"] = brightness_set.get(
        "actual", quiesce_extra.get("display_brightness")
    )
    quiesce_extra["charging_complete"] = complete
    quiesce_extra["charging_complete_reason"] = complete_reason
    quiesce_extra["battery_status_wmi"] = {
        "charging": batt_wmi.charging,
        "discharging": batt_wmi.discharging,
        "charge_rate_mw": batt_wmi.charge_rate_mw,
        "discharge_rate_mw": batt_wmi.discharge_rate_mw,
        "remaining_capacity_mwh": batt_wmi.remaining_capacity_mwh,
        "voltage_mv": batt_wmi.voltage_mv,
        "power_online": batt_wmi.power_online,
    }

    all_cpus = set(_expected_cpus(CONFIGS["A0a"], platform_cfg))
    blocks, seed = schedule_blocks(list(CONFIGS), n_blocks=args.blocks, seed=args.seed)

    report: dict[str, Any] = {
        "spec_path": spec_path.as_posix(),
        "platform": args.platform,
        "config_threads": {cid: c.inference_num_threads for cid, c in CONFIGS.items()},
        "blocks": args.blocks,
        "shuffle_seed": seed,
        "schedule": blocks,
        "cooldown_s": args.cooldown_s,
        "min_phase_samples_declared": _MIN_PHASE_SAMPLES,
        "thermal": {
            "regime": "confound",
            "cooldown_mode": "time_based_unvalidated",
            "note": (
                "Cooldown is time-based; without a temperature ceiling it is unvalidated. "
                "Throttle detection uses frequency (PDH), within-cell drift, and block-position "
                "regression - not package temperature."
            ),
        },
        "quiesce": {
            "pinned_profile": "ac-pinned",
            "on_battery": power.on_battery,
            "battery_pct_start": power.battery_pct,
            "charging": power.charging,
            "charging_complete": complete,
            "charging_complete_reason": complete_reason,
            "power_source": "battery" if power.on_battery else "mains",
            "power_plan_name": power.power_plan_name,
            "power_plan_guid": power.power_plan_guid,
            "overlay_guid": power.overlay_guid,
            "battery_saver": power.battery_saver,
            **quiesce_extra,
        },
        "forbidden_process_check": {
            "patterns": list(forbidden_patterns),
            "hits": forbidden_hits,
            "hard_refuse_hits": hard_hits,
        },
        "runs": {cid: [] for cid in CONFIGS},
        "summary": {},
    }

    out_path = args.out or (root / "derived" / "mslice" / "affinity_matrix.json")
    checkpoint_path = out_path.with_name(out_path.stem + ".partial.json")

    skip_cells: set[tuple[int, str]] = set()
    if args.resume:
        if not checkpoint_path.exists():
            print(f"REFUSED: --resume but checkpoint missing: {checkpoint_path}")
            return 2
        prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for key in ("blocks", "shuffle_seed", "schedule", "config_threads"):
            if prior.get(key) != report.get(key):
                print(
                    f"REFUSED: --resume mismatch on {key}: "
                    f"checkpoint={prior.get(key)!r} current={report.get(key)!r}"
                )
                return 2
        report["runs"] = prior.get("runs") or report["runs"]
        report["resumed_from"] = checkpoint_path.as_posix()
        skip_cells = _completed_cells(report["runs"])
        print(f"Resuming: {len(skip_cells)} cells already scored; skipping those.")
        report.pop("interrupted", None)

    awake_flags = _keep_awake_windows()
    if awake_flags is not None:
        report["quiesce"]["keep_awake"] = "SetThreadExecutionState(SYSTEM|AWAYMODE)"
        print("keep-awake: SetThreadExecutionState SYSTEM_REQUIRED|AWAYMODE_REQUIRED")

    interrupted: dict[str, Any] | None = None
    try:
        for block_index, block in enumerate(blocks):
            print(f"\n=== block {block_index + 1}/{len(blocks)} ===")
            for cell_index, config_id in enumerate(block):
                if (block_index, config_id) in skip_cells:
                    print(f"\n--- {config_id}: skip (checkpoint) ---")
                    continue
                power_now = capture_power_state()
                batt_now = capture_battery_status_wmi()
                if power_now.on_battery and not args.allow_battery:
                    interrupted = {
                        "reason": "ac_lost_mid_matrix",
                        "at_block": block_index,
                        "at_cell": config_id,
                        "battery_pct": power_now.battery_pct,
                        "charging": power_now.charging,
                    }
                    print(
                        "REFUSED mid-matrix: AC lost "
                        f"(block {block_index + 1}, next cell {config_id}, "
                        f"battery_pct={power_now.battery_pct}). Checkpoint preserved. STOP."
                    )
                    break
                still_complete, still_reason = is_charging_complete(
                    power_now,
                    batt_now,
                    charge_rate_max_mw=(
                        float(charge_rate_max) if charge_rate_max is not None else None
                    ),
                    charging_complete_soc_pct=(
                        float(soc_complete) if soc_complete is not None else None
                    ),
                )
                if not still_complete and not args.allow_charging:
                    interrupted = {
                        "reason": "charging_resumed_mid_matrix",
                        "at_block": block_index,
                        "at_cell": config_id,
                        "detail": still_reason,
                        "battery_pct": power_now.battery_pct,
                        "charging": power_now.charging,
                        "charge_rate_mw": batt_now.charge_rate_mw,
                    }
                    print(
                        "ABORT mid-matrix: charging resumed "
                        f"(block {block_index + 1}, next cell {config_id}, "
                        f"{still_reason}). Checkpoint preserved. STOP."
                    )
                    break
                cfg = CONFIGS[config_id]
                print(
                    f"\n--- {config_id}: {cfg.description} "
                    f"(threads={cfg.inference_num_threads}) ---"
                )
                outcome = _spawn(config_id, spec_path=spec_path, platform=args.platform)
                outcome["block"] = block_index
                outcome["cell_index"] = cell_index
                report["runs"][config_id].append(outcome)
                cell_ok = _run_is_ok(outcome)
                gens = outcome.get("generations") or []
                decode = gens[-1].get("r_decode_tok_s") if gens else None
                n_pref = gens[-1].get("n_prefill_util_samples") if gens else None
                print(f"  ok={cell_ok} decode_tok_s={decode} n_prefill_samples={n_pref}")
                if outcome.get("error"):
                    print(f"  ERROR: {outcome['error']}")
                    if outcome.get("returncode") in (4294967295, -1):
                        interrupted = {
                            "reason": "child_killed",
                            "at_block": block_index,
                            "at_cell": config_id,
                            "returncode": outcome.get("returncode"),
                            "stderr_tail": outcome.get("stderr"),
                        }
                        print("Child killed (likely OOM/sleep/AC). Checkpointing and STOP.")
                        _write_checkpoint(checkpoint_path, report)
                        break
                _write_checkpoint(checkpoint_path, report)
                if cell_index < len(block) - 1 or block_index < len(blocks) - 1:
                    print(f"  cooldown {args.cooldown_s}s …")
                    time.sleep(args.cooldown_s)
            if interrupted:
                break

        if interrupted:
            report["interrupted"] = interrupted
            report["quiesce"]["battery_pct_end"] = capture_power_state().battery_pct
            _write_checkpoint(checkpoint_path, report)
            refuse_path = out_path.with_name(out_path.stem + "_interrupted.json")
            refuse_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"wrote interrupted artifact {refuse_path}")
            print(f"wrote checkpoint {checkpoint_path}")
            return 4
    finally:
        _release_awake_windows(awake_flags)

    # TWO-PASS ANALYSIS: all cells collected above; thresholds from A5/A6; then classify.
    pooled = _pool_baselines(report["runs"])
    noise_band = compute_noise_band(pooled)
    p_cpus = {int(c) for c in platform_cfg.get("topology.p_cpus") or ()}
    lpe_cpus = {int(c) for c in platform_cfg.get("topology.lpe_cpus") or ()}
    a5_ok = [r for r in report["runs"].get("A5", []) if _run_is_ok(r)]
    a6_ok = [r for r in report["runs"].get("A6", []) if _run_is_ok(r)]
    a5_decode = _aggregate_deltas(a5_ok, "decode_delta_pct_per_cpu")
    a6_decode = _aggregate_deltas(a6_ok, "decode_delta_pct_per_cpu")
    thr_map = per_core_loaded_thresholds(a5_decode, a6_decode, p_cpus=p_cpus, lpe_cpus=lpe_cpus)
    report["noise_band"] = {
        "value": round(noise_band, 4),
        "formula": "2 * mean(per-core CV of pooled idle baseline means)",
        "pooled_baseline_n_per_core": {str(k): len(v) for k, v in pooled.items()},
    }
    report["loaded_threshold"] = {
        "per_core": {str(k): round(v, 4) for k, v in sorted(thr_map.items())},
        "formula": (
            "threshold(c) = 0.5 * delta(c) in the cell that deliberately targets c; "
            "P-cores from A5 decode deltas; LP-E from A6 decode deltas"
        ),
        "a5_decode_delta_pct_per_cpu": {str(k): round(v, 4) for k, v in sorted(a5_decode.items())},
        "a6_decode_delta_pct_per_cpu": {str(k): round(v, 4) for k, v in sorted(a6_decode.items())},
        "p_cpus": sorted(p_cpus),
        "lpe_cpus": sorted(lpe_cpus),
        "applied_to": "every cell, both prefill and decode (two-pass; not during collection)",
        # Scalar for gate presence checks / legacy printers.
        "value": round(statistics.fmean(thr_map.values()), 4) if thr_map else 0.0,
    }

    for config_id in CONFIGS:
        report["summary"][config_id] = _summarize_cell(
            config_id,
            report["runs"][config_id],
            noise_band=noise_band,
            all_cpus=all_cpus,
            platform_cfg=platform_cfg,
            loaded_thresholds=thr_map,
        )

    report["decision"] = _mechanism_decision(report["summary"])
    report["throttle_detectors"] = _throttle_detectors(report)
    power_end = capture_power_state()
    batt_end = capture_battery_status_wmi()
    report["quiesce"]["battery_pct_end"] = power_end.battery_pct
    report["quiesce"]["charging_end"] = power_end.charging
    report["quiesce"]["battery_status_wmi_end"] = {
        "charging": batt_end.charging,
        "charge_rate_mw": batt_end.charge_rate_mw,
    }

    a0a_diag = report["summary"].get("A0a", {}).get("reference_diagnostics") or {}
    report["decision"]["a0a_diagnostics"] = a0a_diag
    report["decision"]["a0b_diagnostics"] = (
        report["summary"].get("A0b", {}).get("reference_diagnostics") or {}
    )

    verification = _verification_gates(report, min_phase_samples=_MIN_PHASE_SAMPLES)
    report["verification_gates"] = verification

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    run_id = _emit_matrix_manifest(
        report=report,
        out_path=out_path,
        platform_cfg=platform_cfg,
        root=root,
        allow_dirty=bool(args.allow_dirty),
        seed=seed,
        n_blocks=args.blocks,
    )
    report["run_id"] = run_id
    sealed_manifest = root / "raw" / run_id / "manifest.json"
    if not sealed_manifest.is_file():
        verification["pass"] = False
        prior = list(verification.get("failures") or [])
        verification["failures"] = [
            *prior,
            f"sealed manifest missing at {sealed_manifest.as_posix()}",
        ]
    else:
        verification["sealed_manifest"] = sealed_manifest.as_posix()
        verification["run_id"] = run_id
    report["verification_gates"] = verification
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for config_id, summ in report["summary"].items():
        cls = summ.get("classification") or {}
        print(
            f"  {config_id}: verdict={cls.get('verdict')} "
            f"prefill={cls.get('prefill')} decode={cls.get('decode')} "
            f"decode_mean={summ.get('r_decode_tok_s_mean')}"
        )
    print(f"\nLOADED_THRESHOLDS_PER_CORE={report['loaded_threshold']['per_core']}")
    print(f"NOISE_BAND={report['noise_band']['value']}")
    print(f"adopted mechanism: {report['decision']['adopted_mechanism']}")
    print(f"reason: {report['decision']['reason']}")
    print(f"mechanism effect (A3 vs A6): {report['decision']['mechanism_effect']}")
    print(f"throttle_detectors: {report['throttle_detectors']}")
    print(f"verification_gates: {verification}")
    print(f"run_id={run_id}")
    print(f"wrote {out_path}")
    if not verification.get("pass"):
        print("VERIFICATION GATES FAILED - STOP. Do not proceed to full matrix / adoption.")
        return 6
    return 0 if report["decision"]["adopted_mechanism"] != "none" else 1


def _set_display_brightness(target_pct: int | None) -> dict[str, Any]:
    """Set display brightness to the pinned target; record target and actual."""
    out: dict[str, Any] = {"target": target_pct, "actual": None, "ok": False}
    if target_pct is None:
        out["error"] = "no_target_declared"
        return out
    try:
        import subprocess as sp

        script = (
            f"$m = Get-CimInstance -Namespace root/WMI -ClassName "
            f"WmiMonitorBrightnessMethods | Select-Object -First 1; "
            f"Invoke-CimMethod -InputObject $m -MethodName WmiSetBrightness "
            f"-Arguments @{{Timeout=1; Brightness={int(target_pct)}}} | Out-Null; "
            f"(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness "
            f"| Select-Object -First 1).CurrentBrightness"
        )
        completed = sp.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            out["actual"] = float(completed.stdout.strip())
            out["ok"] = abs(out["actual"] - float(target_pct)) <= 1.0
        else:
            out["error"] = (completed.stderr or completed.stdout or "set_failed")[-500:]
    except Exception as exc:
        out["error"] = type(exc).__name__
    return out


def _capture_quiesce_extras() -> dict[str, Any]:
    """Brightness, Defender realtime, ambient - record by value; null when unavailable."""
    out: dict[str, Any] = {
        "display_brightness": None,
        "defender_realtime": None,
        "ambient_c": None,
        "ambient_method": None,
    }
    try:
        import subprocess as sp

        bright = sp.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness "
                "| Select-Object -First 1 -ExpandProperty CurrentBrightness)",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if bright.returncode == 0 and bright.stdout.strip():
            out["display_brightness"] = float(bright.stdout.strip())
    except Exception as exc:
        out["display_brightness_error"] = type(exc).__name__

    try:
        import subprocess as sp

        def_rt = sp.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-MpPreference).DisableRealtimeMonitoring",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if def_rt.returncode == 0 and def_rt.stdout.strip():
            disabled = def_rt.stdout.strip().lower() in {"true", "1"}
            out["defender_realtime"] = "disabled" if disabled else "enabled"
    except Exception as exc:
        out["defender_realtime_error"] = type(exc).__name__

    out["ambient_c"] = None
    out["ambient_method"] = "unavailable_pending_M2.3_LHM"
    out["package_temp_c"] = None
    out["throttle_detection"] = "frequency_based_authorized"
    return out


def _verification_gates(report: dict[str, Any], *, min_phase_samples: int) -> dict[str, Any]:
    """Part-2 gates. Failure is STOP - never descope."""
    failures: list[str] = []
    prefill_ok = True
    for cid, runs in (report.get("runs") or {}).items():
        for run in runs:
            if not _run_is_ok(run):
                prefill_ok = False
                failures.append(f"{cid}/block{run.get('block')}: run not ok")
                continue
            for gen in run.get("generations") or []:
                n_p = int(gen.get("n_prefill_util_samples") or 0)
                n_d = int(gen.get("n_decode_util_samples") or 0)
                if n_p < min_phase_samples or n_d < min_phase_samples:
                    prefill_ok = False
                    failures.append(
                        f"{cid}/block{run.get('block')}/gen{gen.get('gen_index')}: "
                        f"n_prefill={n_p} n_decode={n_d} < {min_phase_samples}"
                    )
                pref_delta = gen.get("prefill_delta_pct_per_cpu") or []
                # Prefill deltas on loaded cores should be positive for saturated work.
                if pref_delta and max(float(v) for v in pref_delta) <= 0:
                    failures.append(
                        f"{cid}/block{run.get('block')}/gen{gen.get('gen_index')}: "
                        "prefill deltas all non-positive"
                    )

    a0a = (report.get("summary") or {}).get("A0a", {}).get("reference_diagnostics") or {}
    a0a_loaded = int(a0a.get("loaded_core_count") or 0)
    # A0a is NOT a saturation gate. Record loaded_core_count as diagnostic only
    # (oversubscription pair A0a vs A0b). Do not fail when A0a loads only 4 P-cores.

    quiesce = report.get("quiesce") or {}
    if quiesce.get("charging") is True or quiesce.get("charging_end") is True:
        failures.append("charging true during block")
    if not quiesce.get("charging_complete", False):
        failures.append("charging_complete false at start")

    lt_block = report.get("loaded_threshold") or {}
    lt_per_core = lt_block.get("per_core")
    nb = (report.get("noise_band") or {}).get("value")
    if not lt_per_core:
        failures.append("per-core LOADED_THRESHOLD map missing (A5/A6 two-pass)")
    if nb is None:
        failures.append("NOISE_BAND missing")

    run_id = report.get("run_id")
    # run_id may be filled after this function; sealed check is separate.
    return {
        "pass": not failures,
        "failures": failures,
        "prefill_windows_ok": prefill_ok,
        "a0a_loaded_core_count": a0a_loaded,
        "a0a_saturation_gate": "demoted_diagnostic_only",
        "loaded_threshold_per_core": lt_per_core,
        "noise_band": nb,
        "charging_complete": quiesce.get("charging_complete"),
        "min_phase_samples": min_phase_samples,
        "run_id_present_at_gate": bool(run_id),
    }


def _emit_matrix_manifest(
    *,
    report: dict[str, Any],
    out_path: Path,
    platform_cfg: Any,
    root: Path,
    allow_dirty: bool,
    seed: int,
    n_blocks: int,
) -> str:
    """Emit a sealed ``raw/<run_id>/`` manifest for the matrix / verification run."""
    from seam.manifest import emit
    from seam.model_provenance import load_local_spec, manifest_model_block
    from seam.powerstate import capture_power_state, manifest_power_state

    spec = load_local_spec(Path(report["spec_path"]))
    power = capture_power_state()
    quiesce = report.get("quiesce") or {}
    model_block = manifest_model_block(
        spec=spec,
        spec_path=Path(report["spec_path"]),
        reasoning_mode="thinking_off",
    )
    # file_verification methods only - drop per-file detail objects that violate schema.
    verification = spec.get("verification") or {}
    file_methods = {
        name: str(entry.get("method"))
        for name, entry in verification.items()
        if isinstance(entry, dict)
        and entry.get("method") in {"sha256", "git-blob-sha1", "size-only"}
    }
    if file_methods:
        model_block["provenance"]["file_verification"] = file_methods

    def _write_outputs(run_dir: Any) -> dict[str, Any]:
        # Copy the derived report into the sealed run so raw_sha256 covers it.
        dest = run_dir.path / "affinity_matrix.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {
            "n_blocks": n_blocks,
            "n_configs": len(CONFIGS),
            "shuffle_seed": seed,
            "loaded_threshold": report.get("loaded_threshold"),
            "noise_band": report.get("noise_band"),
            "adopted_mechanism": (report.get("decision") or {}).get("adopted_mechanism"),
            "verification_gates": report.get("verification_gates"),
            "derived_path": out_path.as_posix(),
        }

    handle = emit(
        config=platform_cfg,
        target="cpu-p",
        workload={
            "kind": "mslice_affinity_matrix",
            "benchmark": "a0a_a0b_a1_a6_confinement_matrix",
            "task_ids": list(CONFIGS),
            "seed": seed,
            "n_repeats": n_blocks,
        },
        condition_label="mslice_affinity_matrix_ac_pinned",
        repo_root=root,
        allow_dirty=allow_dirty,
        summary={
            "matrix_cells": list(CONFIGS),
            "cooldown_s": report.get("cooldown_s"),
            "thermal_regime": "confound",
            "decision": report.get("decision"),
        },
        model=model_block,
        power_state=manifest_power_state(
            power,
            battery_pct_end=quiesce.get("battery_pct_end"),
            display_brightness=quiesce.get("display_brightness"),
            defender_realtime=quiesce.get("defender_realtime"),
        ),
        thermal={"regime": "confound", "excluded": False},
        before_integrity_hash=_write_outputs,
        self_check="pass" if (report.get("verification_gates") or {}).get("pass") else "fail",
    )
    return handle.run_id


def _throttle_detectors(report: dict[str, Any]) -> dict[str, Any]:
    """Three detectors: (a) PDH frequency, (b) within-cell gen1 vs gen2, (c) block-position."""
    freq_methods: set[str] = set()
    freq_min_pct: list[float] = []
    within_cell: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []

    for cid, runs in (report.get("runs") or {}).items():
        for run in runs:
            for gen in run.get("generations") or []:
                freq = (gen.get("sampler_overhead") or {}).get("frequency") or {}
                method = freq.get("method")
                if method:
                    freq_methods.add(str(method))
                summary = freq.get("summary") or {}
                for v in summary.get("min_pct_of_max_per_cpu") or []:
                    if isinstance(v, (int, float)):
                        freq_min_pct.append(float(v))
            gens = run.get("generations") or []
            if len(gens) >= 2:
                g1 = gens[0].get("r_decode_tok_s")
                g2 = gens[1].get("r_decode_tok_s")
                if g1 and g2:
                    drift = (float(g2) - float(g1)) / float(g1)
                    within_cell.append(
                        {
                            "config_id": cid,
                            "block": run.get("block"),
                            "decode_rel_drift_g2_vs_g1": round(drift, 4),
                        }
                    )
            if gens and run.get("block") is not None:
                block_rows.append(
                    {
                        "config_id": cid,
                        "block": int(run["block"]),
                        "decode_tok_s": statistics.fmean(
                            float(g["r_decode_tok_s"])
                            for g in gens
                            if g.get("r_decode_tok_s") is not None
                        ),
                    }
                )

    slope = None
    if len(block_rows) >= 4:
        xs = [r["block"] for r in block_rows]
        ys = [r["decode_tok_s"] for r in block_rows]
        x_mean = statistics.fmean(xs)
        y_mean = statistics.fmean(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = (num / den) if den else None

    flagged = []
    if freq_min_pct and min(freq_min_pct) < 80.0:
        flagged.append("frequency_dip_below_80pct_of_max")
    large_drift = [d for d in within_cell if abs(d["decode_rel_drift_g2_vs_g1"]) > 0.15]
    if large_drift:
        flagged.append("within_cell_decode_drift_gt_15pct")
    if slope is not None and abs(slope) > 0.5:
        flagged.append("block_position_slope_gt_0.5_tok_s_per_block")

    return {
        "a_pdh_frequency": {
            "methods_seen": sorted(freq_methods),
            "n_min_pct_observations": len(freq_min_pct),
            "global_min_pct_of_max": min(freq_min_pct) if freq_min_pct else None,
        },
        "b_within_cell_drift": {
            "n_pairs": len(within_cell),
            "pairs": within_cell,
            "n_flagged_gt_15pct": len(large_drift),
        },
        "c_block_position": {
            "n_rows": len(block_rows),
            "slope_decode_tok_s_per_block": slope,
        },
        "flagged": flagged,
        "excluded_cells": [],
        "note": (
            "Cooldown is time-based and unvalidated without a temperature ceiling. "
            "Flagged cells are recorded; exclusion requires operator confirmation."
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
