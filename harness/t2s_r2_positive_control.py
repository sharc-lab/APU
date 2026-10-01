"""Driver for R2's positive-control detection check (Problem 3): runs run_positive_control() for real,
once per runtime (ollama num_ctx=8192, llama_server -c 8192), and prints each result's
truncation_detected_turn so a caller can verify the canary-based detector actually fires before R2's
real run is unblocked.

Not queue-chained (no tq.advance() call) -- run directly, synchronously, outside the queue, per
t2s_r2_session_growth.py's own module docstring WARNING.

Usage: py -3.12 t2s_r2_positive_control.py --host evo-x2 --model llama31-8b
"""
from __future__ import annotations

import argparse
import json
import sys

import t2s_r2_session_growth as r2
import host_config as hc_mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, choices=["evo-t2s", "evo-x2", "EVO-T2S", "EVO-X2"])
    ap.add_argument("--model", default="llama31-8b")
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args(argv)

    host_key = hc_mod.ALIASES.get(args.host.lower(), args.host.upper())
    host_cfg = hc_mod.HOSTS[host_key]

    results = {}
    for runtime in ("ollama", "llama_server"):
        print(f"=== positive control: runtime={runtime} model={args.model} host={host_key} ===", flush=True)
        try:
            result = r2.run_positive_control(runtime=runtime, model_id=args.model, host_cfg=host_cfg,
                                              max_turns=args.max_turns)
            results[runtime] = result
            # phase_run_session returns {"skipped", "session", "rows", "scored"} -- truncation_detected_turn
            # and loaded_context_tokens live under "scored", not at the top level (bug found live
            # 2026-10-01: this driver's first version read result.get(...) directly and always printed
            # None even when detection had genuinely fired, e.g. the llama_server leg at turn 10).
            scored = result.get("scored", {})
            fired_turn = scored.get("truncation_detected_turn")
            loaded_ctx = scored.get("loaded_context_tokens")
            print(f"{runtime}: loaded_context_tokens={loaded_ctx} truncation_detected_turn={fired_turn}",
                  flush=True)
            print(json.dumps(result, indent=2, default=str), flush=True)
        except Exception as e:
            import traceback
            print(f"{runtime}: ERROR {e!r}", flush=True)
            traceback.print_exc()
            results[runtime] = {"error": repr(e)}

    print("\n=== SUMMARY ===", flush=True)
    for runtime, result in results.items():
        scored = result.get("scored", {}) if isinstance(result, dict) else {}
        fired = scored.get("truncation_detected_turn")
        loaded_ctx = scored.get("loaded_context_tokens")
        print(f"{runtime}: loaded_context_tokens={loaded_ctx} truncation_detected_turn={fired}", flush=True)
    return results


if __name__ == "__main__":
    main()
