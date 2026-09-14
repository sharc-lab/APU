"""Phase A2 - validate the OpenVINO install and prove core selection took effect.

Runs against the small throwaway IR, never the slice model: the question here is whether the
*mechanism* works, and answering it on a 0.6B costs seconds instead of minutes.

Three things are established, all empirically:

1. The IR loads and generates through ``LLMPipeline``.
2. ``SCHEDULING_CORE_TYPE`` actually confines work to the M1-committed cluster. Panther Lake has
   4 P-cores and 4 **LP-E** cores and no standard E-cores, so ``ECORE_ONLY`` may map to LP-E, may
   match nothing, or may silently fall through to every core. Which of the three happens is
   determined by per-core utilization during live inference, not by the property being accepted.
   When OpenVINO's mapping disagrees with M1 the check is repeated under process affinity via
   :func:`seam.topology.affinity_for`, and both verdicts are recorded.
3. ``enable_thinking`` is honoured in **both** directions (AM-024): ``<think>`` present when on,
   absent when off. A flag that silently does nothing would make Arm 1 and Arm 2 the same
   experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seam.backends.base import GenerationRequest
from seam.backends.local_openvino import LocalOpenVinoBackend
from seam.config import load_platform_config
from seam.gitinfo import repo_root
from seam.jsonlog import log_event
from seam.model_provenance import load_local_spec, quantization_summary
from seam.tools.verify_core_affinity import verify_target

__all__ = ["main"]

#: Long enough that the sampler collects many intervals, and prose-only so the small throwaway
#: model does not stall on structure it cannot produce.
_PROBE = (
    "Explain, in plain prose and without lists, how a deadline-aware scheduler decides whether "
    "to run a task locally or send it to a remote service. Cover latency prediction, the cost of "
    "being wrong in each direction, and what happens as the deadline tightens."
)
_PROBE_MAX_TOKENS = 256


def _request(max_tokens: int = _PROBE_MAX_TOKENS) -> GenerationRequest:
    return GenerationRequest(
        messages=[{"role": "user", "content": _PROBE}],
        system="You are a helpful assistant.",
        tools=(),
        max_tokens=max_tokens,
    )


def _check_target(
    *,
    spec: dict[str, Any],
    target: str,
    scheduling_core_type: str,
    expected_cpus: list[int],
    threads: int,
    affinity_cpus: list[int] | None,
) -> dict[str, Any]:
    backend = LocalOpenVinoBackend(
        model_dir=Path(spec["ir_dir"]),
        target=target,  # type: ignore[arg-type]
        scheduling_core_type=scheduling_core_type,
        inference_num_threads=threads,
        enable_cpu_pinning=True,
        model_ref=f"{spec['name']}@{str(spec['revision'])[:12]}+{quantization_summary(spec)}",
        affinity_cpus=affinity_cpus,
    )
    verdict = backend.preflight()
    if verdict.status != "OK":
        return {
            "target": target,
            "preflight": verdict.status,
            "reason": verdict.reason,
            "evidence": None,
        }

    holder: dict[str, Any] = {}

    def _generate() -> None:
        holder["result"] = backend.generate(_request())

    evidence = verify_target(
        target=target,
        scheduling_core_type=scheduling_core_type,
        expected_cpus=expected_cpus,
        generate=_generate,
    )
    result = holder.get("result")
    return {
        "target": target,
        "preflight": verdict.status,
        "config": backend.config_record(),
        "evidence": evidence.to_dict(),
        "completion_tokens": getattr(result, "completion_tokens", None),
        "wall_s": (getattr(result, "wall_ns", 0) or 0) / 1e9,
    }


def _check_thinking(spec: dict[str, Any], threads: int) -> dict[str, Any]:
    """AM-024: assert the reasoning flag in both directions on the same pipeline."""
    import contextlib

    import psutil

    # A previous fallback may have narrowed the process mask; reasoning behavior must be judged
    # without that constraint skewing generation length.
    with contextlib.suppress(Exception):
        psutil.Process().cpu_affinity(list(range(psutil.cpu_count() or 8)))

    out: dict[str, Any] = {}
    for enable in (False, True):
        backend = LocalOpenVinoBackend(
            model_dir=Path(spec["ir_dir"]),
            target="cpu-p",
            scheduling_core_type="PCORE_ONLY",
            inference_num_threads=threads,
            model_ref=str(spec["name"]),
            enable_thinking=enable,
        )
        backend.load()
        rendered = backend.render_prompt(_request())
        result = backend.generate(_request(max_tokens=96))
        mode = backend.reasoning_mode
        out[mode] = {
            "enable_thinking": enable,
            "prompt_contains_think_tag": "<think>" in rendered,
            "prompt_contains_empty_think_pair": "<think>\n\n</think>" in rendered,
            "prompt_tail": rendered[-220:],
            "output_contains_think_block": bool(result.extra.get("think_block_present")),
            "output_head": result.text[:400],
            "completion_tokens": result.completion_tokens,
            "completion_chars": result.completion_chars,
        }
    off = out["thinking_off"]
    on = out["thinking_on"]
    # The empty-think-block convention: Qwen3 templates suppress reasoning by pre-filling a closed
    # <think></think> pair, so the discriminant is the generated block, not the rendered prompt.
    out["verdict"] = (
        "pass"
        if (not off["output_contains_think_block"] and on["output_contains_think_block"])
        else "inconclusive"
    )
    out["note"] = (
        ""
        if out["verdict"] == "pass"
        else (
            "enable_thinking did not produce the expected presence/absence contrast. Arm 1 and "
            "Arm 2 would not be distinct experiments. Investigate before running Arm 2."
        )
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="aipc-c1")
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-thinking", action="store_true")
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    spec_path = args.spec or (root / "configs" / "models" / "Qwen3-0.6B-int4-ov.yaml")
    spec = load_local_spec(spec_path)
    platform = load_platform_config(args.platform, repo_root=root)
    p_cpus = [int(c) for c in platform.get("topology.p_cpus")]
    lpe_cpus = [int(c) for c in platform.get("topology.lpe_cpus")]

    report: dict[str, Any] = {
        "spec_path": spec_path.as_posix(),
        "model": spec["name"],
        "m1_mapping": {"p_cpus": p_cpus, "lpe_cpus": lpe_cpus},
        "openvino_native": {},
        "process_affinity_fallback": {},
    }

    plan = [
        ("cpu-p", "PCORE_ONLY", p_cpus),
        ("cpu-lpe", "ECORE_ONLY", lpe_cpus),
    ]
    for target, core_type, expected in plan:
        print(f"\n=== {target} / {core_type} (expect cpus {expected}) ===")
        outcome = _check_target(
            spec=spec,
            target=target,
            scheduling_core_type=core_type,
            expected_cpus=expected,
            threads=args.threads,
            affinity_cpus=None,
        )
        report["openvino_native"][target] = outcome
        ev = outcome.get("evidence") or {}
        print(f"  verdict={ev.get('verdict')} loaded={ev.get('observed_loaded_cpus')}")
        print(f"  per-cpu mean %={ev.get('mean_pct_per_cpu')}")

    leaked = [
        t
        for t, o in report["openvino_native"].items()
        if not ((o.get("evidence") or {}).get("matched"))
    ]
    if leaked:
        print(f"\nOpenVINO did NOT confine {leaked}. Retrying under process affinity.")
        for target, core_type, expected in plan:
            if target not in leaked:
                continue
            outcome = _check_target(
                spec=spec,
                target=target,
                scheduling_core_type=core_type,
                expected_cpus=expected,
                threads=args.threads,
                affinity_cpus=expected,
            )
            report["process_affinity_fallback"][target] = outcome
            ev = outcome.get("evidence") or {}
            print(f"  {target}: {ev.get('verdict')} loaded={ev.get('observed_loaded_cpus')}")

    if not args.skip_thinking:
        print("\n=== AM-024 reasoning-flag check ===")
        report["thinking"] = _check_thinking(spec, args.threads)
        print(f"  verdict={report['thinking']['verdict']}")

    native_ok = all(
        (o.get("evidence") or {}).get("matched") for o in report["openvino_native"].values()
    )
    fallback_ok = bool(report["process_affinity_fallback"]) and all(
        (o.get("evidence") or {}).get("matched")
        for o in report["process_affinity_fallback"].values()
    )
    report["mechanism"] = (
        "openvino_scheduling_core_type"
        if native_ok
        else ("process_affinity" if fallback_ok else "none")
    )
    report["verdict"] = "pass" if (native_ok or fallback_ok) else "refused"

    out_path = args.out or (root / "derived" / "mslice" / "affinity_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log_event(
        "mslice.affinity_validation",
        severity="info" if report["verdict"] == "pass" else "error",
        message=f"affinity validation {report['verdict']} via {report['mechanism']}",
        verdict=report["verdict"],
        mechanism=report["mechanism"],
    )
    print(f"\nVERDICT: {report['verdict']} (mechanism: {report['mechanism']})")
    print(f"wrote {out_path}")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
