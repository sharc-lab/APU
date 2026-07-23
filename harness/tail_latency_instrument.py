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
    OPENAI_API_KEY=sk-... python tail_latency_instrument.py

Outputs:
    tail_latency_results.json

Output schema per record:
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

import concurrent.futures
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Import shared constants and helpers from the main adapter
from harness.adapters.sdk_direct import (
    BACKEND,
    MODEL,
    PAYLOAD_PROFILE,
    TASKS,
    TOOL_DEFINITIONS,
    TOOL_IMPLS,
    _get_env,
    _get_git_info,
    _get_setup_ref,
)
from harness.instrumentation import wall_ns

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_SAMPLES = 30          # probes per (task, condition) — minimum for stable percentiles
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "tail_latency_results.json"
PROBE_MAX_TOKENS = 512    # keep probes cheap; we care about timing, not output length

# Narrow this list to reduce API cost during development
CHARACTERIZE_TASKS = list(TASKS.keys())

CONDITIONS = ["single", "chained", "fan_out"]

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
# Probe helpers — shared message builder
# ---------------------------------------------------------------------------

def _system_msg() -> dict:
    return {"role": "system", "content": "You are a helpful assistant. Use the provided tools when appropriate."}


def _call_api(client: OpenAI, messages: list[dict]) -> tuple[object, float]:
    """Make one API call, return (response, wall_ms)."""
    t0 = wall_ns()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        max_tokens=PROBE_MAX_TOKENS,
    )
    return response, (wall_ns() - t0) / 1e6


def _execute_tool_calls(response) -> tuple[list[dict], float]:
    """
    Execute all tool calls in a response. Returns (tool_result_messages, total_tool_ms).
    tool_ms is the sum of local execution time across all tool calls in this response.
    """
    tool_results: list[dict] = []
    total_tool_ms = 0.0

    if response.choices[0].finish_reason not in ("tool_calls", "function_call"):
        return tool_results, total_tool_ms

    msg = response.choices[0].message
    if not msg.tool_calls:
        return tool_results, total_tool_ms

    for tc in msg.tool_calls:
        tool_input = json.loads(tc.function.arguments)
        impl       = TOOL_IMPLS.get(tc.function.name, lambda _: f"unknown tool: {tc.function.name}")

        t0 = wall_ns()
        result_str = impl(tool_input)
        total_tool_ms += (wall_ns() - t0) / 1e6

        tool_results.append({
            "role":         "tool",
            "tool_call_id": tc.id,
            "content":      result_str,
        })

    return tool_results, total_tool_ms


# ---------------------------------------------------------------------------
# Probe: single (c=1)
# ---------------------------------------------------------------------------

def _probe_single(client: OpenAI, task_id: str) -> dict[str, float]:
    """
    One complete turn: send task prompt, optionally execute one round of tool calls.

    mcp_roundtrip_ms  — wall time of the single API call
    tool_dispatch_ms  — time spent executing tool implementations locally (0 if no tools)
    turn_total_ms     — end-to-end wall time of the entire probe
    """
    messages = [_system_msg(), {"role": "user", "content": TASKS[task_id]["prompt"]}]

    t_start = wall_ns()
    response, mcp_ms = _call_api(client, messages)
    _, tool_ms = _execute_tool_calls(response)
    turn_ms = (wall_ns() - t_start) / 1e6

    return {"mcp_roundtrip_ms": mcp_ms, "tool_dispatch_ms": tool_ms, "turn_total_ms": turn_ms}


# ---------------------------------------------------------------------------
# Probe: chained (c=3 sequential)
# ---------------------------------------------------------------------------

def _probe_chained(client: OpenAI, task_id: str) -> dict[str, float]:
    """
    Three sequential tool-use turns in one conversation.

    Metrics are summed across all turns:
      mcp_roundtrip_ms  — sum of all API call wall times
      tool_dispatch_ms  — sum of all local tool execution times
      turn_total_ms     — wall time from first call to last tool result appended
    """
    prompt = (
        f"{TASKS[task_id]['prompt']}\n\n"
        "Please use at least one tool in each of your first three replies."
    )
    messages = [_system_msg(), {"role": "user", "content": prompt}]

    t_start      = wall_ns()
    total_mcp_ms  = 0.0
    total_tool_ms = 0.0

    for _ in range(3):
        response, mcp_ms = _call_api(client, messages)
        total_mcp_ms += mcp_ms

        finish = response.choices[0].finish_reason
        if finish == "stop":
            break

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_unset=False))

        tool_results, tool_ms = _execute_tool_calls(response)
        total_tool_ms += tool_ms

        if not tool_results:
            break
        messages.extend(tool_results)

    turn_ms = (wall_ns() - t_start) / 1e6
    return {"mcp_roundtrip_ms": total_mcp_ms, "tool_dispatch_ms": total_tool_ms, "turn_total_ms": turn_ms}


# ---------------------------------------------------------------------------
# Probe: fan_out (c=3 parallel)
# ---------------------------------------------------------------------------

def _probe_fanout(client: OpenAI, task_id: str) -> dict[str, float]:
    """
    Three independent API calls dispatched in parallel (fan-out pattern).

    Latency semantics follow the critical-path model:
      mcp_roundtrip_ms  — max of the three individual call latencies (blocking path)
      tool_dispatch_ms  — sum across all branches (total CPU committed)
      turn_total_ms     — wall time from thread pool submit to last future resolved
    """
    messages = [_system_msg(), {"role": "user", "content": TASKS[task_id]["prompt"]}]

    def _one_call(_: int) -> tuple[float, float]:
        response, mcp_ms = _call_api(client, messages)
        _, tool_ms = _execute_tool_calls(response)
        return mcp_ms, tool_ms

    t_start = wall_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_one_call, i) for i in range(3)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    turn_ms = (wall_ns() - t_start) / 1e6

    return {
        "mcp_roundtrip_ms": max(r[0] for r in results),   # critical path
        "tool_dispatch_ms": sum(r[1] for r in results),   # total CPU expended
        "turn_total_ms":    turn_ms,
    }


# ---------------------------------------------------------------------------
# Probe dispatch table
# ---------------------------------------------------------------------------

_PROBES = {
    "single":  _probe_single,
    "chained": _probe_chained,
    "fan_out": _probe_fanout,
}

# ---------------------------------------------------------------------------
# Sample collector
# ---------------------------------------------------------------------------

def collect(
    client: OpenAI,
    task_id: str,
    condition: str,
    n: int = MIN_SAMPLES,
) -> list[dict]:
    """
    Run n probes for (task_id, condition).
    Returns three tail_stats records: tool_dispatch_ms, mcp_roundtrip_ms, turn_total_ms.
    """
    probe = _PROBES[condition]

    tool_dispatch_samples: list[float] = []
    mcp_roundtrip_samples: list[float] = []
    turn_total_samples:    list[float] = []

    for i in range(n):
        print(f"    run {i + 1:3d}/{n}  {condition}/{task_id}")
        try:
            m = probe(client, task_id)
            tool_dispatch_samples.append(m["tool_dispatch_ms"])
            mcp_roundtrip_samples.append(m["mcp_roundtrip_ms"])
            turn_total_samples.append(m["turn_total_ms"])
        except Exception as exc:
            print(f"      WARNING: probe failed — {exc}")

    return [
        _tail_record(tool_dispatch_samples, task_id, condition, "tool_dispatch_ms"),
        _tail_record(mcp_roundtrip_samples, task_id, condition, "mcp_roundtrip_ms"),
        _tail_record(turn_total_samples,    task_id, condition, "turn_total_ms"),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    all_records: list[dict] = []

    for task_id in CHARACTERIZE_TASKS:
        for condition in CONDITIONS:
            print(f"\n{'─' * 52}")
            print(f"  Task: {task_id}  |  Condition: {condition}")
            print(f"{'─' * 52}")
            records = collect(client, task_id, condition, MIN_SAMPLES)
            all_records.extend(records)

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

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"Output written to {OUTPUT_PATH}   ({len(all_records)} records)")
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
