"""
Live one-shot probe to verify TTFT measurement.

Calls _call_ollama_streaming directly (the same code path used in production sweeps)
with a short prompt guaranteed to produce several tokens, and prints ttft_ms,
latency_ms, tokens_out, and ttft_source.

This is a DIAGNOSTIC ONLY — not an instrumented experiment. Results must not be
recorded as sweep data. NOT a harness result row. NOT cached. NOT conformant with
PROTOCOL.md. No thinking_chars field is captured here.

Cold-start warning
------------------
If the model is not already loaded in Ollama, the first call will load it from disk
into VRAM. On a 2.5 GB Q4_K_M model this takes ~169 s on an RTX 4070 Laptop. That
load time appears as TTFT because the first content token cannot arrive until loading
completes. It is NOT a reasoning/thinking phase. On the first run of this script the
169 s latency was incorrectly attributed to silent thinking; the correct diagnosis is
cold-start model load. Always warm the model with a throwaway call before measuring
TTFT, or discard the first result explicitly.

Usage
-----
    py -3.12 scripts/probe_ttft_live.py [--host http://localhost:11434] [--model qwen3:4b-instruct]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Pull _call_ollama_streaming from harness without importing the full runner
# (which triggers config loading). Add repo root to sys.path first.
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

import harness.runner as _runner  # noqa: E402 (after sys.path manipulation)


PROMPT = (
    "List the first ten prime numbers, one per line. "
    "Output only the numbers, nothing else."
)
MAX_TOKENS = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    args = parser.parse_args()

    print(f"Model : {args.model}")
    print(f"Host  : {args.host}")
    print(f"Prompt: {PROMPT!r}")
    print()

    t_wall_start = time.perf_counter()
    try:
        output, latency_ms, ttft_ms, tokens_in, tokens_out, done_reason, thinking_chars = (
            _runner._call_ollama_streaming(
                args.model, PROMPT, MAX_TOKENS, args.host
            )
        )
    except Exception as exc:
        print(f"ERROR: call failed — {exc}", file=sys.stderr)
        sys.exit(1)
    t_wall_elapsed = (time.perf_counter() - t_wall_start) * 1000

    # Determine ttft_source as run_cell would
    ttft_source = "streamed" if ttft_ms is not None else "streamed-no-token"

    print("Raw results from _call_ollama_streaming:")
    print(f"  latency_ms    : {latency_ms:.1f} ms")
    print(f"  ttft_ms       : {ttft_ms!r}" + (f" ({ttft_ms:.1f} ms)" if ttft_ms else ""))
    print(f"  tokens_in     : {tokens_in}")
    print(f"  tokens_out    : {tokens_out}")
    print(f"  done_reason   : {done_reason!r}")
    print(f"  thinking_chars: {thinking_chars}  (>0 means model thought despite suppress flag)")
    print(f"  output        : {repr(output[:120])}")
    print()
    print("Derived:")
    print(f"  ttft_source : {ttft_source!r}")
    if ttft_ms is not None:
        ratio = ttft_ms / latency_ms if latency_ms > 0 else float("nan")
        print(f"  ttft/total  : {ratio:.3f}  (high ratios normal for thinking models)")
    print(f"  wall_elapsed: {t_wall_elapsed:.1f} ms  (includes Python overhead outside the function)")
    print()

    # Verdict
    if ttft_ms is None:
        print("VERDICT: FAIL — ttft_ms is None; no content token detected.")
        print("  chunk['message']['content'] was empty for every chunk.")
        print("  The fix converted wrong-value (latency_ms) to honest-None, but")
        print("  TTFT is still unmeasured. Check endpoint format and chunk shape.")
        sys.exit(1)
    elif tokens_out == 0:
        print("VERDICT: INCONCLUSIVE — ttft_ms measured but tokens_out=0.")
        print("  No completion tokens counted. Response may be a single non-streaming chunk.")
        sys.exit(1)
    elif ttft_ms == latency_ms:
        print("VERDICT: FAIL — ttft_ms == latency_ms exactly.")
        print("  This is the old fallback artifact. Token detection did not fire;")
        print("  ttft_ms was substituted with latency_ms by the caller.")
        sys.exit(1)
    else:
        # NOTE on thinking models: qwen3 (and similar) run a long silent
        # thinking phase where chunk['message']['content'] is empty.
        # Thinking tokens appear in a separate field (not 'content'), so
        # ttft correctly records the first *answer* token, not the first
        # thinking token.  A ratio close to 1.0 is normal for thinking models
        # (e.g. 169s thinking / 172s total → ratio 0.984).  What matters is
        # that ttft_ms is NOT equal to latency_ms, and that output is non-empty.
        gap_ms = latency_ms - ttft_ms
        print("VERDICT: OK — ttft_ms is a genuine measurement (≠ latency_ms).")
        print(f"  Time-to-first-answer-token: {ttft_ms:.0f} ms")
        print(f"  Time to output rest of response: {gap_ms:.0f} ms")
        print("  Token detection is working correctly on the current /api/chat path.")
        if ratio > 0.95:
            print(f"  (High ratio {ratio:.3f} expected for a thinking model — long silent")
            print("  thinking phase before first answer token; this is not the fallback.)")
        sys.exit(0)


if __name__ == "__main__":
    main()
