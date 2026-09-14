"""One ΔN ladder repeat, isolated in its own process.

Process isolation is the safety property that makes the ceiling search survivable. A rung at
or past the ceiling exhausts memory, and the failure arrives as an allocation throw, an access
violation, or the OS killing the process outright. A child takes only its own repeat with it;
the parent records the failure mode and the ladder continues. It also makes teardown between
rungs total rather than best-effort, since process exit returns everything the allocator was
holding.

Nothing here touches ``raw/``. The child reports JSON on a path the parent chose, and the
parent owns every sealed artifact.

Output is rewritten after each phase so that a child which is killed mid-generation still
leaves behind the phase it reached. A missing field is therefore evidence, not a gap.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import psutil

MB = 1024.0 * 1024.0


def _mem() -> dict[str, float]:
    virtual = psutil.virtual_memory()
    return {
        "rss_mb": psutil.Process().memory_info().rss / MB,
        "available_mb": virtual.available / MB,
        "used_mb": virtual.used / MB,
        "percent": virtual.percent,
    }


class _Writer:
    """Rewrite the result file after every phase so a killed child still reports progress."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.state: dict[str, Any] = {
            "phase": "start",
            "completed": False,
            "failure_mode": None,
            "exception": None,
        }

    def update(self, **fields: Any) -> None:
        self.state.update(fields)
        self.flush()

    def flush(self) -> None:
        tmp = self._path.with_suffix(".partial")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    writer = _Writer(args.out)

    # Baseline is taken before any OpenVINO import so the reservation attributed to a backend
    # includes the runtime it drags in, which is part of what enabling that backend costs.
    writer.update(phase="baseline", memory_baseline=_mem(), spec=spec)

    try:
        return _run(spec, writer)
    except BaseException as exc:
        writer.update(
            phase="failed",
            completed=False,
            failure_mode=f"exception:{type(exc).__name__}",
            exception={
                "type": type(exc).__name__,
                "message": str(exc)[:4000],
                "traceback": traceback.format_exc()[:8000],
            },
            memory_at_failure=_mem(),
        )
        return 1


def _run(spec: dict[str, Any], writer: _Writer) -> int:
    from seam.tools._winpower import assert_system_required

    # Hold PowerRequestSystemRequired for the life of this measurement child so Modern
    # Standby cannot suspend the repeat. Cleared on every exit path, including exceptions.
    power_req = assert_system_required(
        reason="SEAM measurement child; PowerRequestSystemRequired for generate window",
        role="measurement_child",
    )
    writer.update(phase="power_request", power_request=dict(power_req.record))

    try:
        return _run_with_power(spec, writer)
    finally:
        power_req.release()
        writer.update(power_request=dict(power_req.record))


def _run_with_power(spec: dict[str, Any], writer: _Writer) -> int:
    from seam.telemetry.rss import RssSampler
    from seam.tools._winmem import lock_working_set, read_working_set_limits

    affinity = spec.get("affinity_cpus")
    if affinity:
        psutil.Process().cpu_affinity([int(cpu) for cpu in affinity])
    writer.update(affinity_readback=[int(c) for c in psutil.Process().cpu_affinity()])

    # Requested before the model is compiled: a hard minimum granted after the weights are
    # already resident does not protect them from having been trimmed on the way in.
    wslock = dict(spec.get("wslock") or {})
    if str(wslock.get("mode")) == "request":
        lock_record = lock_working_set(
            minimum_bytes=int(wslock["minimum_bytes"]),
            maximum_bytes=int(wslock["maximum_bytes"]),
        )
    else:
        lock_record = {
            "requested": False,
            "granted": False,
            "mode": str(wslock.get("mode", "off")),
            "before": read_working_set_limits(),
        }
    writer.update(phase="wslock", working_set_lock=lock_record)

    import openvino as ov
    import openvino_genai as ov_genai

    from seam.ov_kv_precision import (
        enforce_kv_cache_precision,
        materialize_pipeline_properties,
        requested_kv_cache_precision,
    )

    writer.update(phase="imported", memory_after_import=_mem())

    model_dir = str(spec["model_dir"])
    loads: list[dict[str, Any]] = []
    pipes: dict[str, Any] = {}
    kv_precision_records: list[dict[str, Any]] = []

    # Load order matters and is declared by the parent: the standing accelerator is compiled
    # first, because "enabled" means resident before the measured work begins.
    # One Core per device load so set_property + LLMPipeline + get_property share sticky state.
    for entry in spec["load_sequence"]:
        device = str(entry["device"])
        properties_raw = dict(entry.get("properties") or {})
        properties = materialize_pipeline_properties(properties_raw)
        requested_kv = requested_kv_cache_precision(properties_raw)
        before = _mem()
        t0 = time.perf_counter()
        core = ov.Core()
        if requested_kv is not None:
            core.set_property(device, {"KV_CACHE_PRECISION": properties["KV_CACHE_PRECISION"]})
        pipes[device] = ov_genai.LLMPipeline(model_dir, device, **properties)
        compile_s = time.perf_counter() - t0
        after = _mem()
        kv_check = enforce_kv_cache_precision(device=device, requested=requested_kv, core=core)
        kv_precision_records.append(kv_check)
        load_rec: dict[str, Any] = {
            "device": device,
            "properties": {
                k: (str(v) if hasattr(v, "to_string") else v) for k, v in properties.items()
            },
            "properties_requested": properties_raw,
            "compile_s": compile_s,
            "memory_before": before,
            "memory_after": after,
            "rss_delta_mb": after["rss_mb"] - before["rss_mb"],
            "available_delta_mb": before["available_mb"] - after["available_mb"],
            "kv_cache_precision": kv_check,
        }
        loads.append(load_rec)
        writer.update(
            phase=f"loaded:{device}",
            loads=list(loads),
            kv_cache_precision_readback=list(kv_precision_records),
        )
        if not kv_check["match"]:
            writer.update(
                phase="failed",
                completed=False,
                failure_mode=kv_check["failure_mode"],
                exception={
                    "type": "KvCachePrecisionMismatch",
                    "message": str(kv_check["failure_mode"]),
                    "traceback": "",
                },
                memory_at_failure=_mem(),
            )
            return 1

    # The standing reservation: everything resident after all backends are enabled and before a
    # single token of the measured prompt has been processed.
    baseline = writer.state["memory_baseline"]
    resident = _mem()
    writer.update(
        phase="resident",
        memory_resident_before_inference=resident,
        standing_reservation={
            "rss_delta_mb": resident["rss_mb"] - baseline["rss_mb"],
            "available_delta_mb": baseline["available_mb"] - resident["available_mb"],
            "basis": (
                "delta from process baseline (pre-import) to all backends compiled and resident, "
                "before any measured generation; rss and system-available disagree when weights "
                "are file-backed or shared, and both are reported without attribution"
            ),
        },
    )

    prompt = Path(spec["prompt_path"]).read_text(encoding="utf-8")
    generate_device = str(spec["generate_device"])
    pipe = pipes[generate_device]

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = int(spec["max_new_tokens"])
    cfg.do_sample = False
    cfg.ignore_eos = True
    # The prompt arrives fully rendered so that every arm sees byte-identical input; letting the
    # pipeline re-template it would make prompt length a function of the backend.
    cfg.apply_chat_template = False

    # Same dual-source TTFT instrument the rest of the project uses; the prefill/decode split
    # is not reimplemented here.
    from seam.backends.local_openvino import _make_ttft_streamer, resolve_ttft_ns

    sampler = RssSampler(interval_s=float(spec.get("rss_interval_s", 0.05)))
    writer.update(phase="generating", memory_before_generate=_mem())

    streamer = _make_ttft_streamer(ov_genai)
    sampler.start()
    t0 = time.perf_counter()
    streamer.t0_ns = time.perf_counter_ns()
    try:
        result = pipe.generate([prompt], cfg, streamer)
    finally:
        wall_s = time.perf_counter() - t0
        window = sampler.stop()

    texts = getattr(result, "texts", None)
    text = str(texts[0]) if texts else str(result)
    metrics = getattr(result, "perf_metrics", None)
    prompt_tokens = None
    completion_tokens = None
    metrics_ttft_ns = None
    if metrics is not None:
        try:
            prompt_tokens = int(metrics.get_num_input_tokens())
            completion_tokens = int(metrics.get_num_generated_tokens())
        except Exception:  # optional metric; absence is recorded, never invented
            prompt_tokens = None
            completion_tokens = None
        try:
            ttft_ms = float(metrics.get_ttft().mean)
            metrics_ttft_ns = int(ttft_ms * 1_000_000) if ttft_ms > 0.0 else None
        except Exception:
            metrics_ttft_ns = None

    ttft_ns, ttft_source = resolve_ttft_ns(metrics_ttft_ns, streamer.ttft_ns)
    prefill_s = (ttft_ns / 1e9) if ttft_ns else None
    r_prefill = None
    r_decode = None
    decode_s = None
    if ttft_ns and prompt_tokens:
        r_prefill = prompt_tokens / (ttft_ns / 1e9)
    if ttft_ns and completion_tokens and completion_tokens > 1:
        decode_s = wall_s - (ttft_ns / 1e9)
        if decode_s > 0:
            r_decode = (completion_tokens - 1) / decode_s
        else:
            decode_s = None

    writer.update(
        phase="done",
        completed=True,
        failure_mode=None,
        generation={
            "wall_s": wall_s,
            "ttft_ns": ttft_ns,
            "ttft_source": ttft_source,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_tok_s": r_decode,
            "r_prefill_tok_s": r_prefill,
            "r_decode_tok_s": r_decode,
            "prompt_tokens_reported": prompt_tokens,
            "completion_tokens_reported": completion_tokens,
            "completion_chars": len(text),
            "text_head": text[:200],
            **window.to_record(),
        },
        memory_after_generate=_mem(),
        working_set_limits_after=read_working_set_limits(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
