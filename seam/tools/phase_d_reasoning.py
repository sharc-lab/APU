"""Phase D - reasoning discriminant on the verified 4B (D1 echo, D2 threshold, D3 both arms).

Requires the confinement mechanism adopted by A0-A6. Unpaid. Stops on either-arm failure.
"""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
from pathlib import Path
from typing import Any

from seam.backends.base import GenerationRequest
from seam.backends.local_openvino import LocalOpenVinoBackend, runtime_info
from seam.gitinfo import repo_root
from seam.model_provenance import load_local_spec, quantization_summary
from seam.reasoning import (
    THINKING_OCCURRED_MIN_NON_WS,
    prompt_echo_status,
    strip_prompt_prefix,
    thinking_occurred,
)
from seam.tools.affinity_matrix import CONFIGS

__all__ = ["main"]

_N_TASKS_PER_ARM = 10
_MAX_NEW_TOKENS = 256


def _tasks(n: int) -> list[str]:
    base = [
        "A job has a 400 ms deadline. Prefill is predicted at 120 ms and decode at 8 ms/token "
        "for 40 tokens. Should the scheduler keep the step local or escalate? Answer in one "
        "short paragraph.",
        "Explain when a local-first policy should escalate even if the local model is accurate.",
        "Compare memory-bandwidth-bound decode to compute-bound prefill for deadline risk.",
        "A tool call returns in 50 ms. How should that change the remaining deadline budget?",
        "Why must both partition arms use the same confinement mechanism?",
        "Give one failure mode of treating thermal as an axis when it was a confound.",
        "What goes wrong if token counts are compared across Claude and a local tokenizer?",
        "Describe the advisory-deadline semantics: what is reported when a step overruns?",
        "Why is prompt caching relevant to the cloud cost model but not to local tok/s?",
        "State the isolation invariant in one sentence, then one consequence if it is violated.",
        "How does a warm KV cache change the prefill term in a latency predictor?",
        "When is batching on-device harmful for a deadline-aware agent step?",
    ]
    return (base * ((n // len(base)) + 1))[:n]


def _build_backend(
    *,
    spec_path: Path,
    mechanism: str,
    cluster: str,
    enable_thinking: bool,
    threads: int = 4,
) -> LocalOpenVinoBackend:
    spec = load_local_spec(spec_path)
    if mechanism == "process-affinity":
        cfg = CONFIGS["A5" if cluster == "p" else "A6"]
        return LocalOpenVinoBackend(
            model_dir=Path(spec["ir_dir"]),
            target="cpu-p" if cluster == "p" else "cpu-lpe",
            scheduling_core_type=None,
            inference_num_threads=threads,
            enable_cpu_pinning=None,
            model_ref=f"{spec['name']}@{str(spec['revision'])[:12]}+{quantization_summary(spec)}",
            enable_thinking=enable_thinking,
            affinity_cpus=list(cfg.affinity_cpus) if cfg.affinity_cpus else None,
        )
    if mechanism == "scheduling-core-type":
        cfg = CONFIGS["A2" if cluster == "p" else "A4"]
        return LocalOpenVinoBackend(
            model_dir=Path(spec["ir_dir"]),
            target="cpu-p" if cluster == "p" else "cpu-lpe",
            scheduling_core_type=cfg.scheduling_core_type,
            inference_num_threads=threads,
            enable_cpu_pinning=True,
            model_ref=f"{spec['name']}@{str(spec['revision'])[:12]}+{quantization_summary(spec)}",
            enable_thinking=enable_thinking,
            affinity_cpus=None,
        )
    raise SystemExit(f"unknown or unadopted mechanism: {mechanism!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mechanism",
        required=True,
        choices=["process-affinity", "scheduling-core-type"],
    )
    parser.add_argument("--cluster", default="p", choices=["p", "lpe"])
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--n", type=int, default=_N_TASKS_PER_ARM)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    spec_path = args.spec or (root / "configs" / "models" / "Qwen3-4B-int4-ov.yaml")
    out_path = args.out or (root / "derived" / "mslice" / "phase_d_reasoning.json")

    nonce = secrets.token_hex(16)  # 32 hex chars >= 16
    d1_prompt = (
        f"Remember this opaque token for the session log: {nonce}. "
        "Now answer in one sentence: what is 2+2?"
    )

    # D1 - prompt echo probe (thinking off).
    backend_off = _build_backend(
        spec_path=spec_path,
        mechanism=args.mechanism,
        cluster=args.cluster,
        enable_thinking=False,
    )
    if backend_off.preflight().status != "OK":
        print("REFUSED: preflight", backend_off.preflight())
        return 2
    backend_off.load()
    d1_req = GenerationRequest(
        messages=[{"role": "user", "content": d1_prompt}],
        system="You are a helpful assistant.",
        tools=(),
        max_tokens=64,
    )
    rendered = backend_off.render_prompt(d1_req)
    d1_result = backend_off.generate(d1_req)
    echo = prompt_echo_status(d1_result.text, rendered, nonce=nonce)
    # Also check raw against the user content alone.
    if not echo["nonce_appears_in_raw"]:
        echo_user = prompt_echo_status(d1_result.text, d1_prompt, nonce=nonce)
        if echo_user["nonce_appears_in_raw"]:
            echo = echo_user

    report: dict[str, Any] = {
        "mechanism": args.mechanism,
        "cluster": args.cluster,
        "thermal": {"regime": "confound"},
        "runtime": {"openvino": runtime_info().openvino, "genai": runtime_info().genai},
        "thinking_occurred_min_non_ws": THINKING_OCCURRED_MIN_NON_WS,
        "d1": {
            **echo,
            "rendered_prompt_chars": len(rendered),
            "raw_chars": len(d1_result.text),
            "raw_excerpt": d1_result.text[:500],
        },
        "d2": {
            "threshold": THINKING_OCCURRED_MIN_NON_WS,
            "justification": (
                "Off-arm empty pair is literally '\\n\\n' (0 non-whitespace). Threshold 10 "
                "clears that construction with margin; real thinking chains are far longer."
            ),
        },
        "d3": {"n_per_arm": args.n, "arm_off": [], "arm_on": []},
    }

    tasks = _tasks(args.n)

    def _run_arm(enable_thinking: bool) -> list[dict[str, Any]]:
        backend = _build_backend(
            spec_path=spec_path,
            mechanism=args.mechanism,
            cluster=args.cluster,
            enable_thinking=enable_thinking,
        )
        backend.load()
        rows: list[dict[str, Any]] = []
        for i, task in enumerate(tasks):
            req = GenerationRequest(
                messages=[{"role": "user", "content": task}],
                system="You are a helpful assistant.",
                tools=(),
                max_tokens=_MAX_NEW_TOKENS,
            )
            rendered_i = backend.render_prompt(req)
            result = backend.generate(req)
            if report["d1"]["pipeline"] == "prompt_plus_completion":
                text, _ = strip_prompt_prefix(result.text, rendered_i)
            else:
                text = result.text
            occurred = thinking_occurred(text)
            rows.append(
                {
                    "task_index": i,
                    "thinking_occurred": occurred,
                    "n_out_tokens": result.completion_tokens,
                    "n_out_chars": len(text),
                    "raw_excerpt": result.text[:800] if i < 2 else None,
                    "metric_text_excerpt": text[:800] if i < 2 else None,
                }
            )
        return rows

    report["d3"]["arm_off"] = _run_arm(False)
    report["d3"]["arm_on"] = _run_arm(True)

    off_ok = all(not r["thinking_occurred"] for r in report["d3"]["arm_off"])
    on_ok = all(r["thinking_occurred"] for r in report["d3"]["arm_on"])
    report["d3"]["arm_off_pass"] = off_ok
    report["d3"]["arm_on_pass"] = on_ok
    report["d3"]["n_out_off"] = [r["n_out_tokens"] for r in report["d3"]["arm_off"]]
    report["d3"]["n_out_on"] = [r["n_out_tokens"] for r in report["d3"]["arm_on"]]
    report["d3"]["n_out_off_mean"] = statistics.fmean(report["d3"]["n_out_off"])
    report["d3"]["n_out_on_mean"] = statistics.fmean(report["d3"]["n_out_on"])

    if not off_ok:
        report["verdict"] = "refused"
        report["note"] = (
            "Arm 1 (thinking off) showed THINKING_OCCURRED. The 4B may be ignoring the empty "
            "pre-fill - investigate chat template application. STOP."
        )
    elif not on_ok:
        report["verdict"] = "refused"
        report["note"] = (
            "Arm 2 (thinking on) showed no THINKING_OCCURRED. Thinking is not being enabled. STOP."
        )
    else:
        report["verdict"] = "pass"
        report["note"] = ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "d1": report["d1"]["pipeline"],
                "out": str(out_path),
            },
            indent=2,
        )
    )
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
