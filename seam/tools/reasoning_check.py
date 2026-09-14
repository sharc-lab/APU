"""Phase D - does ``enable_thinking`` actually control the model, in both directions?

Three questions, answered empirically on the slice model:

D1. **Does OpenVINO GenAI echo the prompt in its output?** This decides how every text metric in
    the slice is computed. It is tested with a nonce planted in the prompt: if the nonce comes
    back, the runtime returns prompt+completion and ``completion_chars`` would be measuring the
    prompt.
D2. **Content, not markers.** The off-arm pre-fills an *empty* ``<think></think>`` pair, so marker
    presence proves nothing. See :mod:`seam.reasoning`.
D3. **Both directions on the 4B**, with raw output captured.

This gates **both** arms. If the model reasons despite the empty pre-fill, Arm 1 is not a
non-reasoning condition and the two arms are the same experiment.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from seam.backends.base import GenerationRequest
from seam.backends.local_openvino import LocalOpenVinoBackend
from seam.gitinfo import repo_root
from seam.jsonlog import log_event
from seam.model_provenance import load_local_spec, quantization_summary
from seam.reasoning import reasoning_verdict, split_reasoning

__all__ = ["main"]

_TASK = (
    "A user asks you to add 47 and 58, then tell them whether the result is prime. "
    "Give the final answer."
)
_MAX_TOKENS = 384


def _backend(
    spec: dict[str, Any], *, enable_thinking: bool, threads: int, affinity: list[int] | None
) -> LocalOpenVinoBackend:
    return LocalOpenVinoBackend(
        model_dir=Path(spec["ir_dir"]),
        target="cpu-p",
        scheduling_core_type=None if affinity else "PCORE_ONLY",
        inference_num_threads=threads,
        enable_cpu_pinning=None,
        model_ref=f"{spec['name']}@{str(spec['revision'])[:12]}+{quantization_summary(spec)}",
        enable_thinking=enable_thinking,
        affinity_cpus=affinity,
    )


def _probe_prompt_echo(backend: LocalOpenVinoBackend) -> dict[str, Any]:
    """D1: plant a nonce in the prompt and see whether the runtime hands it back."""
    nonce = f"NONCE{uuid.uuid4().hex[:12].upper()}"
    request = GenerationRequest(
        messages=[{"role": "user", "content": f"Reply with the single word OK. Marker: {nonce}"}],
        system="You are a helpful assistant.",
        tools=(),
        max_tokens=24,
    )
    rendered = backend.render_prompt(request)
    result = backend.generate(request)
    echoed = nonce in result.text
    return {
        "nonce_in_rendered_prompt": nonce in rendered,
        "nonce_in_output": echoed,
        "returns_prompt_plus_completion": echoed,
        "output_text": result.text,
        "output_chars": len(result.text),
        "rendered_prompt_chars": len(rendered),
        "interpretation": (
            "Runtime ECHOES the prompt. completion_chars/bytes must be computed on the "
            "post-prompt remainder, or every text metric in the slice is dominated by prompt "
            "length."
            if echoed
            else "Runtime returns the COMPLETION ONLY. completion_chars/bytes are safe as-is."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--affinity", default="", help="comma-separated cpus, e.g. 0,1,2,3")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    spec_path = args.spec or (root / "configs" / "models" / "Qwen3-4B-int4-ov.yaml")
    spec = load_local_spec(spec_path)
    affinity = [int(c) for c in args.affinity.split(",") if c.strip()] or None

    report: dict[str, Any] = {"spec_path": spec_path.as_posix(), "model": spec["name"]}

    print("=== D1: does the runtime echo the prompt? ===")
    probe_backend = _backend(spec, enable_thinking=False, threads=args.threads, affinity=affinity)
    probe_backend.load()
    report["d1_prompt_echo"] = _probe_prompt_echo(probe_backend)
    print(f"  nonce_in_output={report['d1_prompt_echo']['nonce_in_output']}")
    print(f"  {report['d1_prompt_echo']['interpretation']}")

    print("\n=== D2/D3: reasoning contrast on the 4B ===")
    texts: dict[str, str] = {}
    for enable in (False, True):
        backend = _backend(spec, enable_thinking=enable, threads=args.threads, affinity=affinity)
        backend.load()
        request = GenerationRequest(
            messages=[{"role": "user", "content": _TASK}],
            system="You are a helpful assistant.",
            tools=(),
            max_tokens=_MAX_TOKENS,
        )
        rendered = backend.render_prompt(request)
        result = backend.generate(request)
        mode = backend.reasoning_mode
        texts[mode] = result.text
        split = split_reasoning(result.text)
        report[mode] = {
            "enable_thinking": enable,
            "prompt_contains_empty_think_pair": "<think>\n\n</think>" in rendered,
            "rendered_prompt_tail": rendered[-160:],
            "raw_output": result.text,
            "split": split.to_dict(),
            "completion_tokens": result.completion_tokens,
            "wall_s": result.wall_ns / 1e9,
        }
        print(
            f"  {mode}: thinking_chars={split.to_dict()['thinking_chars']} "
            f"tokens={result.completion_tokens} wall={result.wall_ns / 1e9:.1f}s"
        )

    report["verdict"] = reasoning_verdict(
        off_text=texts["thinking_off"], on_text=texts["thinking_on"]
    )
    out_path = args.out or (root / "derived" / "mslice" / "reasoning_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    verdict = str(report["verdict"]["verdict"])
    log_event(
        "mslice.reasoning_check",
        severity="info" if verdict == "pass" else "error",
        message=f"reasoning discriminant {verdict}",
        verdict=verdict,
        prompt_echoed=report["d1_prompt_echo"]["nonce_in_output"],
    )
    print(f"\nVERDICT: {verdict}")
    if report["verdict"]["note"]:
        print(f"  {report['verdict']['note']}")
    print(f"wrote {out_path}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
