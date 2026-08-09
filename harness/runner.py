"""Quality-degradation sweep runner.

Sweeps every probe in evaluation/probes/prompts.jsonl across a range of filler
context depths, with N repetitions per (probe, depth) cell. Uses
evaluation/probes/scorers.py for all scoring so there is one canonical scorer
shared between validation and the sweep.

Usage:
    py -3.11 -m harness.runner [options]
    py -3.11 -m harness.runner --probe-ids rea_01 str_01 --reps 2
    py -3.11 -m harness.runner --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from harness import cache, context, telemetry
from harness.context import DEFAULT_FILLER_MODE, FILLER_MODES

REPO_ROOT = Path(__file__).parent.parent
PROBES_DIR = REPO_ROOT / "evaluation" / "probes"
RESULTS_DIR = REPO_ROOT / "results"

CONTEXT_DEPTHS = [0, 2_000, 8_000, 16_000, 32_000, 64_000]
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_REPS = 5


def _load_scorers():
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:12]


def _call_ollama_streaming(
    model: str, prompt: str, max_tokens: int, host: str
) -> tuple[str, float, float, int, int]:
    """Return (text, latency_ms, ttft_ms, tokens_in, tokens_out)."""
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    start = time.perf_counter()
    ttft_ms: float | None = None
    full_text = ""
    tokens_in = 0
    tokens_out = 0

    with httpx.stream("POST", url, json=payload, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            raw = raw.strip()
            if not raw:
                continue
            chunk = json.loads(raw)
            token = chunk.get("response", "")
            if token and ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000
            full_text += token
            if chunk.get("done"):
                tokens_in = chunk.get("prompt_eval_count", 0)
                tokens_out = chunk.get("eval_count", 0)
                break

    latency_ms = (time.perf_counter() - start) * 1000
    return full_text, latency_ms, ttft_ms or latency_ms, tokens_in, tokens_out


def run_cell(
    probe: dict[str, Any],
    depth: int,
    rep: int,
    model: str,
    host: str,
    cfg_hash: str,
    filler_mode: str,
    scorers,
) -> dict[str, Any]:
    filler = context.build_filler(depth, seed=rep)
    prompt = context.wrap_prompt(filler, probe["prompt"], filler_mode=filler_mode)
    max_tokens: int = probe["max_tokens"]
    params = {"max_tokens": max_tokens, "temperature": 0, "filler_mode": filler_mode}

    cached = cache.get(model, prompt, params)
    if cached:
        output = cached["output"]
        tel = telemetry.Telemetry.from_dict(cached["telemetry"])
    else:
        output, latency_ms, ttft_ms, tokens_in, tokens_out = _call_ollama_streaming(
            model, prompt, max_tokens, host
        )
        tel = telemetry.Telemetry(
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            mem_rss_mb=telemetry.rss_mb(),
            gpu_mem_mb=telemetry.gpu_mem_mb(),
        )
        cache.put(model, prompt, params, {"output": output, "telemetry": tel.to_dict()})

    score_val, score_detail = scorers.score(probe, output)

    return {
        "probe_id": probe["id"],
        "category": probe["category"],
        "difficulty": probe["difficulty"],
        "depth": depth,
        "rep": rep,
        "filler_mode": filler_mode,
        "score": score_val,
        "score_detail": score_detail,
        "latency_ms": round(tel.latency_ms, 1),
        "ttft_ms": round(tel.ttft_ms, 1),
        "tokens_in": tel.tokens_in,
        "tokens_out": tel.tokens_out,
        "mem_rss_mb": round(tel.mem_rss_mb, 1),
        "gpu_mem_mb": round(tel.gpu_mem_mb, 1),
        "config_hash": cfg_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-degradation context sweep")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument(
        "--depths", nargs="+", type=int, default=CONTEXT_DEPTHS,
        metavar="N", help="Token depths for filler (default: 0 2000 8000 16000 32000 64000)",
    )
    parser.add_argument(
        "--probe-ids", nargs="*", metavar="ID",
        help="Run only these probe IDs; default runs all",
    )
    parser.add_argument(
        "--evalset", default=str(PROBES_DIR / "prompts.jsonl"),
        help="Path to prompts.jsonl",
    )
    parser.add_argument(
        "--skip-judge", action="store_true", default=True,
        help="Skip judge-scored probes (default: True)",
    )
    parser.add_argument(
        "--filler-mode", default=DEFAULT_FILLER_MODE, choices=FILLER_MODES,
        help=(
            "unlabelled (default): filler prepended with no framing, measures "
            "context degradation. labelled: filler in <background_context> tags "
            "with ignore instruction, measures instruction-following instead."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sweep plan and exit without calling the model",
    )
    args = parser.parse_args()

    scorers = _load_scorers()

    cfg = {
        "model": args.model,
        "host": args.host,
        "reps": args.reps,
        "depths": sorted(args.depths),
        "filler_mode": args.filler_mode,
    }
    cfg_hash = _config_hash(cfg)

    probes: list[dict] = []
    with open(args.evalset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))

    if args.probe_ids:
        probes = [p for p in probes if p["id"] in args.probe_ids]
    if args.skip_judge:
        probes = [p for p in probes if p["scorer_type"] != "judge"]

    total = len(probes) * len(args.depths) * args.reps

    print(f"Sweep : {len(probes)} probes x {len(args.depths)} depths x {args.reps} reps = {total} calls")
    print(f"Model : {args.model}  Host: {args.host}")
    print(f"Filler: {args.filler_mode}")
    print(f"Config: {cfg_hash}")

    if args.dry_run:
        print("\n[dry-run] exiting before any model calls.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"run_{ts}.jsonl"
    print(f"Output: {out_path}\n")

    done = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for probe in probes:
            for depth in sorted(args.depths):
                for rep in range(args.reps):
                    try:
                        row = run_cell(
                            probe, depth, rep,
                            args.model, args.host, cfg_hash,
                            args.filler_mode, scorers,
                        )
                    except Exception as exc:
                        row = {
                            "probe_id": probe["id"],
                            "category": probe["category"],
                            "depth": depth,
                            "rep": rep,
                            "filler_mode": args.filler_mode,
                            "score": None,
                            "error": str(exc),
                            "config_hash": cfg_hash,
                        }

                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    done += 1

                    score_str = (
                        f"{row['score']:.3f}" if row.get("score") is not None else "ERR"
                    )
                    lat = row.get("latency_ms", 0)
                    w = len(str(total))
                    print(
                        f"[{done:{w}}/{total}] "
                        f"{probe['id']} d={depth:>5} r={rep} "
                        f"score={score_str} lat={lat:.0f}ms"
                    )

    print(f"\nDone. Results -> {out_path}")


if __name__ == "__main__":
    main()
