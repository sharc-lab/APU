#!/usr/bin/env python3
"""
tail_latency_instrument.py

Tail-latency profiler that wraps claude_code_adapter to capture p50/p95/p99
distributions for tool dispatch, LLM round-trip, and total turn latency
under three concurrency conditions.

Conditions:
  single   — one independent API call per probe (c=1)
  chained  — three sequential tool-use turns in one conversation (c=3 serial)
  fan_out  — three parallel API calls fired simultaneously (c=3 parallel)

Usage:
    OPENAI_API_KEY=sk-... python -m harness.tail_latency_instrument
    python -m harness.tail_latency_instrument --resume       # continue interrupted run
    python -m harness.tail_latency_instrument --no-fsync     # skip disk sync (dev)

Durability:
    Each sample is written immediately to JSONL_PATH as it completes
    (fsync per row, truncated-final-line tolerance, config-hash guard).
    The final JSON at OUTPUT_PATH is computed from all samples after
    the sweep completes and is byte-identical to a full un-interrupted run
    for the same samples.

Output files:
    results/tail_latency_results.json   — final aggregate (gitignored)
    results/tail_latency_results.jsonl  — per-sample resume log (gitignored)

Output schema per JSON record:
    {
        "condition":  "single" | "chained" | "fan_out",
        "task_id":    string,
        "metric":     "tool_dispatch_ms" | "mcp_roundtrip_ms" | "turn_total_ms",
        "p50":        number,
        "p95":        number,
        "p99":        number,
        "n":          number,
        "samples":    number[]
    }
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Import shared constants and helpers from the main adapter
from harness.adapters.sdk_direct import (
    BACKEND,
    MODEL,
    REPLAY_MODE,
    PAYLOAD_PROFILE,
    TASKS,
    TOOL_DEFINITIONS,
    TOOL_IMPLS,
    TRACES_ROOT_ENV,
    OpenAIChatBackend,
    _get_env,
    _get_git_info,
    _get_setup_ref,
)
from harness.instrumentation import wall_ns
from harness.replay import ReplayCache

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_SAMPLES = 30          # probes per (task, condition) — minimum for stable percentiles
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "tail_latency_results.json"
JSONL_PATH  = Path(__file__).parent.parent / "results" / "tail_latency_results.jsonl"
PROBE_MAX_TOKENS = 512    # keep probes cheap; we care about timing, not output length

CHARACTERIZE_TASKS = list(TASKS.keys())
CONDITIONS = ["single", "chained", "fan_out"]

# Config hash covers all dimensions that define the output structure.
# Changing model, sample count, task list, or conditions requires a fresh run.
_CFG_SRC = json.dumps(
    {"model": MODEL, "min_samples": MIN_SAMPLES,
     "tasks": sorted(CHARACTERIZE_TASKS), "conditions": CONDITIONS},
    sort_keys=True,
)
CONFIG_HASH = hashlib.sha256(_CFG_SRC.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Percentile calculation
# ---------------------------------------------------------------------------

def _pct(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s   = sorted(samples)
    n   = len(s)
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _tail_record(samples: list[float], task_id: str, condition: str, metric: str) -> dict:
    return {
        "condition": condition,
        "task_id":   task_id,
        "metric":    metric,
        "p50":       _pct(samples, 50),
        "p95":       _pct(samples, 95),
        "p99":       _pct(samples, 99),
        "n":         len(samples),
        "samples":   samples,
    }


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _system_msg() -> dict:
    return {"role": "system", "content": "You are a helpful assistant. Use the provided tools when appropriate."}


def _call_api(backend: OpenAIChatBackend, messages: list[dict]) -> tuple[dict, float, float]:
    result = backend.model_call(
        model=MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        temperature=None,
        seed=None,
        tool_choice="auto",
        max_tokens=PROBE_MAX_TOKENS,
    )
    return result.response_json, result.recorded_latency_ms, result.replay_latency_ms


def _execute_tool_calls(response: dict) -> tuple[list[dict], float]:
    tool_results: list[dict] = []
    total_tool_ms = 0.0

    if response["choices"][0].get("finish_reason") not in ("tool_calls", "function_call"):
        return tool_results, total_tool_ms

    msg = response["choices"][0]["message"]
    if not msg.get("tool_calls"):
        return tool_results, total_tool_ms

    for tc in msg["tool_calls"]:
        tool_input = json.loads(tc["function"]["arguments"])
        tool_name = tc["function"]["name"]
        impl = TOOL_IMPLS.get(tool_name, lambda _: f"unknown tool: {tool_name}")

        t0 = wall_ns()
        result_str = impl(tool_input)
        total_tool_ms += (wall_ns() - t0) / 1e6

        tool_results.append({
            "role":         "tool",
            "tool_call_id": tc["id"],
            "content":      result_str,
        })

    return tool_results, total_tool_ms


# ---------------------------------------------------------------------------
# Probe functions
# ---------------------------------------------------------------------------

def _probe_single(backend: OpenAIChatBackend, task_id: str, sample_index: int = 0) -> dict[str, float]:
    messages = [_system_msg(), {"role": "user", "content": TASKS[task_id]["prompt"]}]
    t_start = wall_ns()
    response, recorded_mcp_ms, replay_mcp_ms = _call_api(backend, messages)
    _, tool_ms = _execute_tool_calls(response)
    turn_ms = (wall_ns() - t_start) / 1e6
    return {
        "mcp_roundtrip_ms":    recorded_mcp_ms,
        "replay_roundtrip_ms": replay_mcp_ms,
        "tool_dispatch_ms":    tool_ms,
        "turn_total_ms":       turn_ms,
    }


def _probe_chained(backend: OpenAIChatBackend, task_id: str, sample_index: int = 0) -> dict[str, float]:
    prompt = (
        f"{TASKS[task_id]['prompt']}\n\n"
        "Please use at least one tool in each of your first three replies."
    )
    messages = [_system_msg(), {"role": "user", "content": prompt}]

    t_start = wall_ns()
    total_mcp_ms = 0.0
    total_replay_mcp_ms = 0.0
    total_tool_ms = 0.0

    for _ in range(3):
        response, mcp_ms, replay_mcp_ms = _call_api(backend, messages)
        total_mcp_ms += mcp_ms
        total_replay_mcp_ms += replay_mcp_ms

        if response["choices"][0].get("finish_reason") == "stop":
            break

        msg = response["choices"][0]["message"]
        messages.append(msg)

        tool_results, tool_ms = _execute_tool_calls(response)
        total_tool_ms += tool_ms

        if not tool_results:
            break
        messages.extend(tool_results)

    turn_ms = (wall_ns() - t_start) / 1e6
    return {
        "mcp_roundtrip_ms":    total_mcp_ms,
        "replay_roundtrip_ms": total_replay_mcp_ms,
        "tool_dispatch_ms":    total_tool_ms,
        "turn_total_ms":       turn_ms,
    }


def _probe_fanout(backend: OpenAIChatBackend, task_id: str, sample_index: int = 0) -> dict[str, float]:
    messages = [_system_msg(), {"role": "user", "content": TASKS[task_id]["prompt"]}]

    def _one_call(_: int) -> tuple[float, float, float]:
        response, mcp_ms, replay_mcp_ms = _call_api(backend, messages)
        _, tool_ms = _execute_tool_calls(response)
        return mcp_ms, replay_mcp_ms, tool_ms

    t_start = wall_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_one_call, i) for i in range(3)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    turn_ms = (wall_ns() - t_start) / 1e6

    return {
        "mcp_roundtrip_ms":    max(r[0] for r in results),
        "replay_roundtrip_ms": max(r[1] for r in results),
        "tool_dispatch_ms":    sum(r[2] for r in results),
        "turn_total_ms":       turn_ms,
    }


_PROBES = {
    "single":  _probe_single,
    "chained": _probe_chained,
    "fan_out": _probe_fanout,
}


# ---------------------------------------------------------------------------
# Resume: load completed samples from JSONL
# ---------------------------------------------------------------------------

def _load_jsonl_samples(
    path: Path,
    cfg_hash: str,
) -> tuple[set[tuple[str, str, int]], dict[tuple[str, str], list[dict]]]:
    """Load completed samples from *path*.

    Returns (completed_set, prior_samples) where:
      completed_set  — {(task_id, condition, sample_index)} already written
      prior_samples  — {(task_id, condition): [sample_dict, ...]}

    Truncated final line (power-loss pattern) is discarded with a warning.
    Malformed non-final line raises ValueError.
    Config hash mismatch raises ValueError.
    """
    completed: set[tuple[str, str, int]] = set()
    prior_samples: dict[tuple[str, str], list[dict]] = defaultdict(list)

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
                f"(not the final line — corruption, not a truncated write): {exc}"
            ) from exc

        existing_hash = row.get("config_hash")
        if existing_hash and existing_hash != cfg_hash:
            raise ValueError(
                f"Config hash mismatch: resume file has {existing_hash!r}, "
                f"current run has {cfg_hash!r}. "
                "Model, task list, condition list, or MIN_SAMPLES changed — "
                "delete the .jsonl file or start a fresh run."
            )

        task_id = row.get("task_id")
        condition = row.get("condition")
        sample_index = row.get("sample_index")

        if task_id and condition and sample_index is not None:
            sample = {
                "mcp_roundtrip_ms":    row.get("mcp_roundtrip_ms"),
                "replay_roundtrip_ms": row.get("replay_roundtrip_ms"),
                "tool_dispatch_ms":    row.get("tool_dispatch_ms"),
                "turn_total_ms":       row.get("turn_total_ms"),
            }
            prior_samples[(task_id, condition)].append(sample)
            completed.add((task_id, condition, sample_index))

    return completed, dict(prior_samples)


# ---------------------------------------------------------------------------
# Sample collector (incremental write)
# ---------------------------------------------------------------------------

def _run_all(
    backend: OpenAIChatBackend,
    out_jsonl: Path,
    completed: set[tuple[str, str, int]],
    prior_samples: dict[tuple[str, str], list[dict]],
    no_fsync: bool,
) -> dict[tuple[str, str], list[dict]]:
    """Run all (task_id, condition, sample_index) triples not already completed.

    Writes each sample to *out_jsonl* immediately after collection.
    Returns a complete sample dict containing both prior and newly collected
    samples for every (task_id, condition) pair.
    """
    all_samples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for k, v in prior_samples.items():
        all_samples[k].extend(v)

    if not no_fsync and not out_jsonl.exists():
        # Fsync the directory before creating the file so the new directory
        # entry is durable. Must run before open() — append mode creates the
        # file immediately, making the exists() check always False inside.
        try:
            fd = os.open(str(out_jsonl.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for task_id in CHARACTERIZE_TASKS:
            for condition in CONDITIONS:
                print(f"\n{'─' * 52}")
                print(f"  Task: {task_id}  |  Condition: {condition}")
                print(f"{'─' * 52}")
                probe = _PROBES[condition]

                for i in range(MIN_SAMPLES):
                    if (task_id, condition, i) in completed:
                        continue

                    print(f"    sample {i + 1:3d}/{MIN_SAMPLES}  {condition}/{task_id}")
                    try:
                        m = probe(backend, task_id, i)
                    except Exception as exc:
                        print(f"      WARNING: probe failed — {exc}")
                        m = {
                            "mcp_roundtrip_ms":    None,
                            "replay_roundtrip_ms": None,
                            "tool_dispatch_ms":    None,
                            "turn_total_ms":       None,
                        }

                    row = {
                        "task_id":      task_id,
                        "condition":    condition,
                        "sample_index": i,
                        "config_hash":  CONFIG_HASH,
                        **m,
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    if not no_fsync:
                        try:
                            os.fsync(fout.fileno())
                        except OSError:
                            pass

                    all_samples[(task_id, condition)].append(m)

    return dict(all_samples)


# ---------------------------------------------------------------------------
# Aggregate computation (unchanged from original)
# ---------------------------------------------------------------------------

def _compute_tail_records(
    all_samples: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    """Compute tail statistics from collected samples.

    Returns the same list structure as the original collect() output.
    For a full run (no prior samples, no failures), the result is
    byte-identical to the original implementation for the same probes.
    """
    records: list[dict] = []
    for task_id in CHARACTERIZE_TASKS:
        for condition in CONDITIONS:
            samples = all_samples.get((task_id, condition), [])

            def _extract(key: str) -> list[float]:
                return [m[key] for m in samples if m.get(key) is not None]

            records.extend([
                _tail_record(_extract("tool_dispatch_ms"),    task_id, condition, "tool_dispatch_ms"),
                _tail_record(_extract("mcp_roundtrip_ms"),    task_id, condition, "mcp_roundtrip_ms"),
                _tail_record(_extract("replay_roundtrip_ms"), task_id, condition, "replay_roundtrip_ms"),
                _tail_record(_extract("turn_total_ms"),       task_id, condition, "turn_total_ms"),
            ])
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            f"Resume an interrupted run. Reads {JSONL_PATH.name} to recover "
            "completed samples; appends new samples to the same file."
        ),
    )
    parser.add_argument(
        "--no-fsync", action="store_true",
        help="Skip os.fsync after each sample write. Faster for dev; unsafe across power loss.",
    )
    args = parser.parse_args()

    completed: set[tuple[str, str, int]] = set()
    prior_samples: dict[tuple[str, str], list[dict]] = {}

    if args.resume:
        if not JSONL_PATH.exists():
            raise FileNotFoundError(
                f"--resume specified but {JSONL_PATH} does not exist. "
                "Start a fresh run without --resume."
            )
        completed, prior_samples = _load_jsonl_samples(JSONL_PATH, CONFIG_HASH)
        print(f"Resume: {len(completed)} samples already done from {JSONL_PATH.name}")

    total = len(CHARACTERIZE_TASKS) * len(CONDITIONS) * MIN_SAMPLES
    remaining = total - len(completed)
    print(f"Config hash : {CONFIG_HASH}")
    print(f"Total       : {total} samples  ({remaining} remaining)")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    traces_root = Path(TRACES_ROOT_ENV) if TRACES_ROOT_ENV else None
    replay_cache = ReplayCache(mode=REPLAY_MODE, traces_root=traces_root)
    backend = OpenAIChatBackend(client=client, replay_cache=replay_cache)

    all_samples = _run_all(
        backend=backend,
        out_jsonl=JSONL_PATH,
        completed=completed,
        prior_samples=prior_samples,
        no_fsync=args.no_fsync,
    )

    all_records = _compute_tail_records(all_samples)

    output = {
        "experiment":   "tail_latency_characterization",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "setup_ref":    _get_setup_ref(),
        "git":          _get_git_info(),
        "env":          _get_env(),
        "config": {
            "model":                    MODEL,
            "backend":                  BACKEND,
            "payload_profile":          PAYLOAD_PROFILE,
            "min_samples_per_condition": MIN_SAMPLES,
            "conditions":               CONDITIONS,
            "tasks":                    CHARACTERIZE_TASKS,
        },
        "results": all_records,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"Output written to {OUTPUT_PATH}   ({len(all_records)} records)")
    print(f"Resume log    at  {JSONL_PATH}")
    print(f"{'=' * 70}")
    print(f"\n{'Task':<8} {'Condition':<10} {'Metric':<22} {'p50':>8} {'p95':>8} {'p99':>8} {'n':>5}")
    print("─" * 70)
    for r in all_records:
        print(
            f"{r['task_id']:<8} {r['condition']:<10} {r['metric']:<22} "
            f"{r['p50']:>8.1f} {r['p95']:>8.1f} {r['p99']:>8.1f} {r['n']:>5}"
        )


if __name__ == "__main__":
    main()
