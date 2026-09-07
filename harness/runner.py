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
import os
import random
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
DEFAULT_MODEL = "qwen3:4b-instruct"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_REPS = 5

# Fields required in every result row written to results/run_*.jsonl.
# memory_architecture distinguishes unified AI PC configs from discrete
# measurement hosts so they are never silently pooled in analysis.
REQUIRED_ROW_FIELDS = frozenset({
    "probe_id", "category", "depth", "rep", "filler_mode",
    "score", "config_hash", "hardware_config", "memory_architecture",
    "model", "model_variant", "thinking_enabled",
    "ctx_suspect", "position_in_cell",
})


def validate_result_row(row: dict) -> None:
    """Raise ValueError if a result row is missing required fields."""
    missing = REQUIRED_ROW_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Result row missing required fields: {sorted(missing)}")


def _load_scorers():
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:12]


def _make_count_fn(host: str, model: str):
    """Return a callable that counts filler tokens via a single-token generation.

    Sends the filler as a user message, requests num_predict=1, and reads
    prompt_eval_count from the done chunk. Subtracts an estimate of the chat
    template overhead (~8 tokens for qwen3 instruct) so the count reflects
    the filler text alone. Callers should treat the result as ±10 tokens
    accurate; the 2% filler tolerance absorbs this.
    """
    TEMPLATE_OVERHEAD = 8

    def count_fn(text: str) -> int:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "stream": True,
            "options": {"num_predict": 1, "temperature": 0},
        }
        with httpx.stream("POST", f"{host}/api/chat", json=payload, timeout=300) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                raw = raw.strip()
                if not raw:
                    continue
                c = json.loads(raw)
                if c.get("done"):
                    return max(0, c.get("prompt_eval_count", 0) - TEMPLATE_OVERHEAD)
        return 0

    return count_fn


def _call_ollama_streaming(
    model: str, prompt: str, max_tokens: int, host: str,
) -> tuple[str, float, float, int, int]:
    """Return (text, latency_ms, ttft_ms, tokens_in, tokens_out).

    Uses /api/chat so the model's chat template is applied. The composed
    filler+probe string is wrapped as a single user message. For the instruct
    model (qwen3:4b-instruct) no think flag is needed — it answers directly.
    """
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
        },
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
            token = chunk.get("message", {}).get("content", "")
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
    filler: str,
    depth: int,
    rep: int,
    position_in_cell: int,
    cell_probe_seed: int,
    model: str,
    host: str,
    cfg_hash: str,
    filler_mode: str,
    hardware_config: str,
    memory_architecture: str,
    model_variant: str,
    thinking_enabled: bool,
    scorers,
) -> dict[str, Any]:
    prompt = context.wrap_prompt(filler, probe["prompt"], filler_mode=filler_mode)
    max_tokens: int = probe["max_tokens"]
    params = {
        "max_tokens": max_tokens,
        "temperature": 0,
        "filler_mode": filler_mode,
        "model_variant": model_variant,
    }

    cached = cache.get(model, prompt, params)
    if cached:
        output = cached["output"]
        tel = telemetry.Telemetry.from_dict(cached["telemetry"])
    else:
        output, latency_ms, ttft_ms, tokens_in, tokens_out = _call_ollama_streaming(
            model, prompt, max_tokens, host,
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

    # Flag rows where context delivered is materially less than requested.
    # tokens_in includes filler + probe + chat template; at depth>0 the filler
    # dominates. A ratio < 0.9 almost certainly indicates a filler undershoot.
    ctx_suspect = depth > 0 and tel.tokens_in < depth * 0.9

    return {
        "probe_id": probe["id"],
        "category": probe["category"],
        "difficulty": probe["difficulty"],
        "depth": depth,
        "rep": rep,
        "position_in_cell": position_in_cell,
        "cell_probe_seed": cell_probe_seed,
        "filler_mode": filler_mode,
        "score": score_val,
        "score_detail": score_detail,
        "latency_ms": round(tel.latency_ms, 1),
        "ttft_ms": round(tel.ttft_ms, 1),
        "tokens_in": tel.tokens_in,
        "tokens_out": tel.tokens_out,
        "max_tokens": max_tokens,
        "ctx_suspect": ctx_suspect,
        "mem_rss_mb": round(tel.mem_rss_mb, 1),
        "gpu_mem_mb": round(tel.gpu_mem_mb, 1),
        "config_hash": cfg_hash,
        "hardware_config": hardware_config,
        "memory_architecture": memory_architecture,
        "model": model,
        "model_variant": model_variant,
        "thinking_enabled": thinking_enabled,
    }


def _load_completed(path: Path, cfg_hash: str) -> set[tuple[int, int, str]]:
    """Return (depth, rep, probe_id) triples already written to *path*.

    File-size note: at the default sweep (44 probes × 6 depths × 5 reps) the
    resume file is ≈0.8 MB; at the largest plausible sweep ≈11 MB.  Loading
    the whole file with read_text() is fine at these sizes and lets us detect
    the last non-empty line without two-pass streaming.

    A malformed FINAL line is treated as a truncated write (the power-loss
    pattern: write + flush reached the OS buffer, fsync did not complete before
    power cut) and is discarded with a warning that names the file and line
    number.  A malformed line at any other position raises — that is real
    corruption, not a truncated write.
    """
    completed: set[tuple[int, int, str]] = set()
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [(i + 1, ln) for i, ln in enumerate(raw_lines) if ln.strip()]
    for idx, (lineno, raw) in enumerate(nonempty):
        is_last = idx == len(nonempty) - 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            if is_last:
                print(
                    f"WARNING: discarding truncated final line in {path} "
                    f"(line {lineno}): {exc}",
                    flush=True,
                )
                continue
            raise ValueError(
                f"Malformed JSON at {path}:{lineno} "
                f"(not the final line — this is corruption, not a truncated write): {exc}"
            ) from exc
        existing_hash = row.get("config_hash")
        if existing_hash and existing_hash != cfg_hash:
            raise ValueError(
                f"Config hash mismatch: resume file has {existing_hash!r}, "
                f"current run has {cfg_hash!r}. Check --depths/--reps/--model."
            )
        if "depth" in row and "rep" in row and "probe_id" in row:
            completed.add((row["depth"], row["rep"], row["probe_id"]))
    return completed


def _fsync_dir(d: Path) -> None:
    """Fsync the directory so a newly-created file's directory entry is durable.

    Required after creating a new result file: os.fsync on the file descriptor
    makes the file contents durable but does not guarantee the directory entry
    (the file's existence) survives a power loss before the directory page is
    written.  Silently no-ops on platforms that do not support opening a
    directory with O_RDONLY (e.g. Windows).
    """
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


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
    parser.add_argument(
        "--hardware-config", default="unknown",
        metavar="NAME",
        help=(
            "Name of the hardware config running this sweep "
            "(e.g. blade14_rtx4070, strix_halo_64gb). Written to every result row "
            "so runs from different hosts cannot be silently pooled."
        ),
    )
    parser.add_argument(
        "--memory-architecture", default="unknown",
        choices=["unified", "discrete", "unknown"],
        help=(
            "Memory architecture of the host: 'unified' (AI PC, weights and KV "
            "share one pool) or 'discrete' (separate VRAM). Written to every "
            "result row. Quality-vs-depth results transfer across architectures; "
            "TTFT and throughput do not."
        ),
    )
    parser.add_argument(
        "--model-variant", default=None, metavar="VARIANT",
        choices=["instruct", "reasoning"],
        help=(
            "Model variant: 'instruct' (answers directly, no chain-of-thought) or "
            "'reasoning' (thinking model). Auto-detected from model name if omitted."
        ),
    )
    parser.add_argument(
        "--resume", default=None, metavar="FILE",
        help=(
            "Resume an interrupted sweep. Reads FILE to find already-completed "
            "(depth, rep, probe_id) triples and skips them; appends new results "
            "to the same FILE. Config hash must match."
        ),
    )
    parser.add_argument(
        "--no-fsync", action="store_true",
        help=(
            "Skip os.fsync after each row write. "
            "Faster for local runs; unsafe across power loss."
        ),
    )
    args = parser.parse_args()

    scorers = _load_scorers()

    model_variant = args.model_variant or (
        "instruct" if "instruct" in args.model.lower() else "reasoning"
    )
    thinking_enabled = model_variant == "reasoning"

    cfg = {
        "model": args.model,
        "model_variant": model_variant,
        "host": args.host,
        "reps": args.reps,
        "depths": sorted(args.depths),
        "filler_mode": args.filler_mode,
        "hardware_config": args.hardware_config,
        "memory_architecture": args.memory_architecture,
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

    # Load already-completed rows when resuming.
    completed: set[tuple[int, int, str]] = set()
    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume file not found: {resume_path}")
        completed = _load_completed(resume_path, cfg_hash)
        print(f"Resume: {len(completed)} rows already done in {resume_path.name}")

    count_fn = _make_count_fn(args.host, args.model)

    print(f"Sweep : {len(probes)} probes x {len(args.depths)} depths x {args.reps} reps = {total} calls")
    print(f"Model : {args.model}  variant={model_variant}  thinking={thinking_enabled}")
    print(f"Host  : {args.host}")
    print(f"Filler: {args.filler_mode}  (count_fn calibration enabled)")
    print(f"HW    : {args.hardware_config}  arch={args.memory_architecture}")
    print(f"Config: {cfg_hash}")

    # Pre-build all unique (depth, rep) fillers with the calibrated count_fn.
    # count_fn fires once per unique pair (not once per probe × depth × rep).
    # These calls are instrumentation: they do not appear in result rows or in
    # latency statistics, which capture only _call_ollama_streaming.
    unique_depth_reps = sorted({(d, r) for d in args.depths for r in range(args.reps)})
    filler_cache: dict[tuple[int, int], str] = {}
    non_zero = [(d, r) for d, r in unique_depth_reps if d > 0]
    if non_zero:
        print(f"\nCalibrating filler for {len(non_zero)} depth×rep pairs "
              f"({len(unique_depth_reps) - len(non_zero)} at d=0 need no calibration)...")
    for d, r in unique_depth_reps:
        fn = count_fn if (d > 0 and not args.dry_run) else None
        filler_cache[(d, r)] = context.build_filler(d, seed=r, count_fn=fn)
    if non_zero:
        print("Filler calibration complete.\n")

    if args.dry_run:
        print("\n[dry-run] exiting before any model calls (filler calibration skipped).")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if resume_path is not None:
        out_path = resume_path
        file_mode = "a"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"run_{ts}.jsonl"
        file_mode = "w"
    remaining = total - len(completed)
    print(f"Output: {out_path}  ({'append' if file_mode == 'a' else 'new'})\n")

    done = 0
    w = len(str(total))
    with open(out_path, file_mode, encoding="utf-8") as fout:
        if file_mode == "w" and not args.no_fsync:
            # Fsync the directory so this new file's existence is durable,
            # not just its contents.  Only needed on the create path; the
            # resume (append) path implies the entry is already committed.
            _fsync_dir(out_path.parent)
        for depth in sorted(args.depths):
            for rep in range(args.reps):
                cell_probe_seed = depth * 100 + rep
                cell_probes = list(probes)
                random.Random(cell_probe_seed).shuffle(cell_probes)
                for pos, probe in enumerate(cell_probes):
                    if (depth, rep, probe["id"]) in completed:
                        done += 1
                        continue
                    try:
                        row = run_cell(
                            probe,
                            filler_cache[(depth, rep)],
                            depth, rep, pos, cell_probe_seed,
                            args.model, args.host, cfg_hash,
                            args.filler_mode,
                            args.hardware_config,
                            args.memory_architecture,
                            model_variant,
                            thinking_enabled,
                            scorers,
                        )
                    except Exception as exc:
                        row = {
                            "probe_id": probe["id"],
                            "category": probe["category"],
                            "depth": depth,
                            "rep": rep,
                            "position_in_cell": pos,
                            "cell_probe_seed": cell_probe_seed,
                            "filler_mode": args.filler_mode,
                            "score": None,
                            "error": str(exc),
                            "config_hash": cfg_hash,
                            "hardware_config": args.hardware_config,
                            "memory_architecture": args.memory_architecture,
                            "model": args.model,
                            "model_variant": model_variant,
                            "thinking_enabled": thinking_enabled,
                            "ctx_suspect": False,
                        }

                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    if not args.no_fsync:
                        os.fsync(fout.fileno())
                    done += 1

                    score_str = (
                        f"{row['score']:.3f}" if row.get("score") is not None else "ERR"
                    )
                    lat = row.get("latency_ms", 0)
                    print(
                        f"[{done:{w}}/{total}] "
                        f"{probe['id']} d={depth:>5} r={rep} pos={pos} "
                        f"score={score_str} lat={lat:.0f}ms",
                        flush=True,
                    )

    print(f"\nDone. Results -> {out_path}")


if __name__ == "__main__":
    main()
