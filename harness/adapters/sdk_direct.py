#!/usr/bin/env python3
"""
claude_code_adapter.py

APU characterization harness using the OpenAI SDK (gpt-4o-mini).
Output schema matches Zachary Johnson's LangGraph harness exactly
(replication_remote_search_v3.json), enabling direct cross-framework comparison.

Usage:
    OPENAI_API_KEY=sk-... python claude_code_adapter.py

Outputs:
    claude_code_characterization.json
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from harness.adapters.base import BackendBase
from harness.instrumentation import (
    Category, CPU_BOUND_CATS,
    HARNESS_STRICT_CATEGORIES, HARNESS_BROAD_CATEGORIES,
    HARNESS_STRICT_DEFINITION, HARNESS_BROAD_DEFINITION,
    wall_ns, process_cpu_ns, Span,
)
from harness.replay import ReplayCache

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
INSTR_VERSION = 3
BACKEND = "openai"
PAYLOAD_PROFILE = "claude_code_adapter"
PROFILE = "mixed"
SEARCH_LOCALITY = "local"   # mock tools — no remote HTTP
MAX_TURNS = 10
N_SEEDS    = int(os.environ.get("N_SEEDS",    "5"))
N_SESSIONS = int(os.environ.get("N_SESSIONS", "10"))
DEBUG      = os.environ.get("SHARC_DEBUG", "0") == "1"
REPLAY_MODE = os.environ.get("APU_REPLAY_MODE", "AUTO")
TRACES_ROOT_ENV = os.environ.get("APU_TRACES_ROOT")
OUTPUT_PATH = Path(__file__).parent.parent.parent / "results" / "claude_code_characterization.json"

# Harness category groupings (from Zachary's definitions)
HARNESS_STRICT_CATEGORIES = frozenset({"ORCH_SETUP", "ORCH_DISPATCH", "TOKENIZATION", "SERIALIZATION"})
HARNESS_BROAD_CATEGORIES  = HARNESS_STRICT_CATEGORIES | frozenset(
    {"HTTP_CLIENT", "PROMPT_ASSEMBLY", "CONTEXT_MGMT", "LOGGING"}
)

HARNESS_STRICT_DEFINITION = "harness_strict (ORCH_SETUP+ORCH_DISPATCH+TOKENIZATION+SERIALIZATION)"
HARNESS_BROAD_DEFINITION  = "harness_broad (strict + HTTP_CLIENT + PROMPT_ASSEMBLY + CONTEXT_MGMT + LOGGING)"
ATTRIBUTION_DOC           = "apu_characterization/ATTRIBUTION.md"

# ---------------------------------------------------------------------------
# Task definitions — 14 tasks, same IDs as Zachary's suite
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "CH-01": {
        "category": "code_hybrid",
        "plan_len": 3,
        "prompt": (
            "Write a Python function that computes Fibonacci numbers with memoization. "
            "Use the calculator tool to verify: fib(10) = 55 and fib(15) = 610. "
            "Then write code to print the first 20 Fibonacci numbers."
        ),
    },
    "CH-02": {
        "category": "code_hybrid",
        "plan_len": 4,
        "prompt": (
            "Search for information about Python decorators and how they work. "
            "Then write a @retry decorator with exponential backoff. "
            "Use the calculator to verify: if base delay is 100ms and multiplier is 2, "
            "what is the delay after 5 retries (in ms)?"
        ),
    },
    "CN-01": {
        "category": "compute_numerical",
        "plan_len": 3,
        "prompt": (
            "Use the calculator tool to compute each of the following: "
            "(1) The sum of squares from 1 to 100. "
            "(2) Verify using n*(n+1)*(2*n+1)/6 with n=100. "
            "(3) The 25th triangular number using n*(n+1)/2."
        ),
    },
    "FO-01": {
        "category": "file_output",
        "plan_len": 3,
        "prompt": (
            "Write a structured comparison of bubble sort, merge sort, and quicksort: "
            "include time complexity (best/average/worst) and space complexity. "
            "Save the result to sorting_algorithms.txt and confirm the file was created."
        ),
    },
    "LH-01": {
        "category": "long_horizon",
        "plan_len": 10,
        "prompt": (
            "Complete this five-step task: "
            "1) Search for REST API design best practices. "
            "2) Write a Python Flask route for a /users GET endpoint with pagination. "
            "3) Use the calculator: if each request takes 45ms, how many requests/second can one worker handle? "
            "4) Write a pytest unit test for the endpoint. "
            "5) Save the route and test together to api_design.txt."
        ),
    },
    "LH-02": {
        "category": "long_horizon",
        "plan_len": 8,
        "prompt": (
            "Complete this five-step task: "
            "1) Search for Python async/await patterns and event loops. "
            "2) Write an async function that fetches data from three sources concurrently. "
            "3) Calculate: if each fetch takes 200ms sequentially vs all three in parallel, what is the speedup ratio? "
            "4) Write the synchronous equivalent for comparison. "
            "5) Save both versions to async_comparison.txt."
        ),
    },
    "RE-01": {
        "category": "retrieval",
        "plan_len": 2,
        "prompt": (
            "Search for information about the transformer neural network architecture. "
            "Provide a comprehensive technical summary covering: self-attention mechanism, "
            "multi-head attention, positional encoding, feed-forward layers, and encoder-decoder structure."
        ),
    },
    "RE-02": {
        "category": "retrieval",
        "plan_len": 2,
        "prompt": (
            "Search for recent developments in large language models. "
            "Summarize the main architectural innovations, training techniques, "
            "and scaling approaches, focusing on technical details."
        ),
    },
    "RH-01": {
        "category": "retrieval_hybrid",
        "plan_len": 4,
        "prompt": (
            "Search for Python coding best practices and style guidelines. "
            "Write a code example demonstrating at least four of the practices found. "
            "Use the calculator: if best practices reduce bugs by 30% and each bug costs 3 hours "
            "to fix, how many hours are saved per 100 bugs?"
        ),
    },
    "RH-02": {
        "category": "retrieval_hybrid",
        "plan_len": 4,
        "prompt": (
            "Search for relational database normalization and the rules for 3NF. "
            "Write a SQL schema example demonstrating first, second, and third normal form. "
            "Calculate: if a denormalized table has 2000000 rows with 40% redundancy, "
            "how many rows does normalization eliminate?"
        ),
    },
    "SH-01": {
        "category": "search_hybrid",
        "plan_len": 4,
        "prompt": (
            "Search for AI research breakthroughs from the past year. "
            "Then search for their practical real-world applications. "
            "Write a concise technical summary of the two most significant findings "
            "and their implications for software development."
        ),
    },
    "SH-02": {
        "category": "search_hybrid",
        "plan_len": 5,
        "prompt": (
            "Search for cloud computing trends and serverless architecture patterns. "
            "Then search for cost comparison data between serverless and container-based hosting. "
            "Write a structured recommendation covering: cost, latency, ops overhead, "
            "and when each approach is appropriate."
        ),
    },
    "SO-01": {
        "category": "search_only",
        "plan_len": 3,
        "prompt": (
            "Search for how quantum computing works, focusing on qubits and superposition. "
            "Then search for quantum computing applications in cryptography and optimization. "
            "Provide a comprehensive summary of both searches."
        ),
    },
    "SW-01": {
        "category": "sweep_canary",
        "plan_len": 3,
        "prompt": (
            "Search for software testing methodologies. "
            "Summarize the key differences between unit testing, integration testing, "
            "and end-to-end testing, and explain when to use each approach."
        ),
    },
}

# ---------------------------------------------------------------------------
# Tool definitions — OpenAI function-calling format
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on any topic. Returns relevant text snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression using +, -, *, /, **, (, )",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write text content to a named file in the adapter_output directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name (no path)"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_exec",
            "description": "Execute a Python code snippet and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_search(query: str) -> str:
    return (
        f"Search results for '{query}':\n"
        f"[1] Overview: {query} encompasses several key concepts widely used in "
        f"modern software development and research communities.\n"
        f"[2] Technical detail: The primary mechanisms involve structured data pipelines, "
        f"algorithmic optimization, and layered abstractions that improve scalability.\n"
        f"[3] Applications: Engineers apply these patterns in production systems to "
        f"improve throughput, reliability, and maintainability at scale."
    )


def _tool_calculator(expression: str) -> str:
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def _tool_file_write(filename: str, content: str) -> str:
    out_dir = Path(__file__).parent.parent / "results" / "adapter_output"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / Path(filename).name  # strip directory traversal
    path.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"


def _tool_code_exec(code: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = (result.stdout or "")[:2000]
        err = (result.stderr or "")[:500]
        return (out + ("\n[stderr] " + err if err else "")).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 10 seconds"
    except Exception as exc:
        return f"Error: {exc}"


TOOL_IMPLS: dict[str, callable] = {
    "search":     lambda inp: _tool_search(inp["query"]),
    "calculator": lambda inp: _tool_calculator(inp["expression"]),
    "file_write": lambda inp: _tool_file_write(inp["filename"], inp["content"]),
    "code_exec":  lambda inp: _tool_code_exec(inp["code"]),
}


class OpenAIChatBackend(BackendBase):
    """OpenAI chat-completions backend that inherits replay behavior from BackendBase."""

    def __init__(self, client: OpenAI, replay_cache: ReplayCache | None = None) -> None:
        super().__init__(replay_cache=replay_cache)
        self.client = client

    def _call_model_api(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
        seed: int | None,
        **kwargs,
    ):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            **kwargs,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        return self.client.chat.completions.create(**payload)

# ---------------------------------------------------------------------------
# Instrumentation primitives
# ---------------------------------------------------------------------------
#
# Windows timing strategy:
#   - time.perf_counter_ns()  → high-res wall clock (~100 ns resolution)
#   - time.process_time_ns()  → process CPU, BUT has 15.6 ms tick on Windows
#                               → only usable for session-level totals, not sub-ms spans
#
# Per-span CPU accounting:
#   CPU-BOUND spans (ORCH_SETUP, TOKENIZATION, etc.): the thread is running
#     Python/JSON work — no blocking I/O. Wall elapsed ≈ CPU consumed.
#     We use wall elapsed as the cpu_ns proxy.
#
#   I/O-BOUND spans (HTTP_CLIENT, CLIENT_HTTP): the thread sleeps on network.
#     cpu_ns = 0 explicitly — we cannot measure it accurately on Windows
#     and it is genuinely near-zero (kernel I/O wait, not user CPU).
#
#   TOOL_COMPUTE: mock tools are pure CPU (string concat, eval, subprocess).
#     Wall elapsed ≈ CPU consumed; use wall proxy.
#
# Session-level thread_cpu_ns: sampled with process_time_ns() across the full
#   session. Coarse (15.6 ms ticks) but accurate over multi-second sessions.
#   RESIDUAL = thread_cpu_ns − sum(cpu_ns of all non-I/O spans).

# Categories where wall time is a valid CPU proxy (no blocking I/O)
CPU_BOUND_CATS = frozenset({
    "ORCH_SETUP", "ORCH_DISPATCH", "TOKENIZATION", "SERIALIZATION",
    "CLIENT_PARSE", "FRAMEWORK", "TOOL_COMPUTE",
})

def wall_ns() -> int:
    """High-resolution monotonic wall clock — ~100 ns on Windows."""
    return time.perf_counter_ns()


def process_cpu_ns() -> int:
    """Process CPU time. 15.6 ms resolution on Windows — session-level only."""
    return time.process_time_ns()





# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def run_session(
    backend: OpenAIChatBackend,
    session_id: str,
    task_id: str,
    task: dict,
) -> dict:
    """
    Run one complete agentic session for a task.

    Returns a Session-shaped dict (Zachary-compatible) plus two internal fields
    prefixed with '_' that are stripped before JSON output:
        _spans   : {category_name: Span}
        _wall_ns : total session wall time in nanoseconds
    """
    spans: dict[str, Span] = defaultdict(Span)

    # OpenAI messages list: system + user, then alternating assistant/tool
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when appropriate to complete tasks accurately."},
        {"role": "user",   "content": task["prompt"]},
    ]

    tool_call_counts: dict[str, int] = {}
    tool_call_sequence: list[dict] = []
    turns = 0
    llm_step = 0
    replayed_call_count = 0
    api_recorded_latency_ms_total = 0.0
    api_replay_latency_ms_total = 0.0

    sess_wall_t0 = wall_ns()
    sess_cpu_t0  = process_cpu_ns()   # coarse 15.6ms ticks, but accurate over the session

    while turns < MAX_TURNS:

        # ── ORCH_SETUP ────────────────────────────────────────────────────
        # JSON-serialize messages to measure payload size + bookkeeping cost.
        # CPU-bound: no I/O. Wall elapsed used as cpu_ns proxy.
        t0 = wall_ns()
        prompt_bytes = len(json.dumps(messages).encode())
        elapsed = wall_ns() - t0
        spans["ORCH_SETUP"].record(elapsed, elapsed)          # cpu = wall proxy

        # ── TOKENIZATION ──────────────────────────────────────────────────
        # Re-encode full payload — measures the serialization cost the SDK pays.
        # CPU-bound: wall proxy.
        t0 = wall_ns()
        _ = json.dumps(messages).encode()
        elapsed = wall_ns() - t0
        spans["TOKENIZATION"].record(elapsed, elapsed, b_in=prompt_bytes)

        # ── HTTP_CLIENT / CLIENT_HTTP ─────────────────────────────────────
        # The API call blocks on network I/O — the thread is sleeping, not
        # burning CPU. cpu_ns = 0 for both: we cannot measure thread CPU
        # accurately on Windows during a blocking syscall, and it is
        # genuinely near-zero (kernel I/O wait, not user-space work).
        # wall_ns = full round-trip time (the meaningful latency signal).
        call_result = backend.model_call(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            temperature=None,
            seed=None,
            tool_choice="auto",
            max_tokens=1024,
        )
        response = call_result.response_json
        if call_result.replayed:
            replayed_call_count += 1
        api_recorded_latency_ms_total += call_result.recorded_latency_ms
        api_replay_latency_ms_total += call_result.replay_latency_ms

        http_wall = int(call_result.recorded_latency_ms * 1e6)
        resp_bytes = len(json.dumps(response).encode())

        spans["HTTP_CLIENT"].record(0, http_wall, b_out=resp_bytes)          # cpu=0: I/O wait
        spans["CLIENT_HTTP"].record(0, http_wall, b_in=prompt_bytes, b_out=resp_bytes)

        turns += 1
        choice = response["choices"][0]
        message = choice["message"]
        finish = choice.get("finish_reason")

        # ── FRAMEWORK ─────────────────────────────────────────────────────
        # Unpack response object, append to message list.
        # CPU-bound: wall proxy.
        t0 = wall_ns()
        messages.append(message)
        elapsed = wall_ns() - t0
        spans["FRAMEWORK"].record(elapsed, elapsed)

        if finish == "stop" or finish not in ("tool_calls", "function_call"):
            break

        if not message.get("tool_calls"):
            break

        # ── Per-tool-call instrumentation ─────────────────────────────────
        tool_result_messages: list[dict] = []

        for tc in message["tool_calls"]:
            tool_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]

            # CLIENT_PARSE: JSON-decode tool arguments. CPU-bound, wall proxy.
            t0 = wall_ns()
            tool_input  = json.loads(raw_args)
            input_bytes = len(raw_args.encode())
            elapsed = wall_ns() - t0
            spans["CLIENT_PARSE"].record(elapsed, elapsed, b_in=input_bytes)

            # TOOL_COMPUTE: mock tools are CPU-bound (string ops, eval).
            # Wall elapsed is a valid CPU proxy here.
            impl = TOOL_IMPLS.get(tool_name, lambda _: f"unknown tool: {tool_name}")
            t0 = wall_ns()
            result_str   = impl(tool_input)
            elapsed      = wall_ns() - t0
            result_bytes = len(result_str.encode())
            spans["TOOL_COMPUTE"].record(elapsed, elapsed, b_in=input_bytes, b_out=result_bytes)

            # SERIALIZATION: pack tool result into message dict. CPU-bound, wall proxy.
            t0 = wall_ns()
            tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result_str}
            serialized_bytes = len(json.dumps(tool_msg).encode())
            elapsed = wall_ns() - t0
            spans["SERIALIZATION"].record(elapsed, elapsed, b_out=serialized_bytes)

            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            tool_call_sequence.append({"tool": tool_name, "query": raw_args, "llm_step": llm_step})
            tool_result_messages.append(tool_msg)

        llm_step += 1

        # ── ORCH_DISPATCH ─────────────────────────────────────────────────
        # Extend message list with tool results. CPU-bound, wall proxy.
        t0 = wall_ns()
        messages.extend(tool_result_messages)
        elapsed = wall_ns() - t0
        spans["ORCH_DISPATCH"].record(elapsed, elapsed)

    sess_wall_t1 = wall_ns()
    sess_cpu_t1  = process_cpu_ns()

    wall_ns_total = sess_wall_t1 - sess_wall_t0
    # process_time_ns delta: coarse (15.6ms ticks) but correct over multi-second sessions
    thread_cpu_ns = max(0, sess_cpu_t1 - sess_cpu_t0)

    # Non-I/O spans: their cpu_ns values are wall-proxy estimates of real CPU work
    non_io_cpu = sum(
        s.cpu_ns for cat, s in spans.items()
        if cat not in ("HTTP_CLIENT", "CLIENT_HTTP")
    )

    # RESIDUAL: CPU the OS saw that our span proxies didn't capture
    # (Python interpreter overhead, GC, SDK glue code between measured points)
    residual_ns = max(0, thread_cpu_ns - non_io_cpu)
    if residual_ns > 0:
        spans["RESIDUAL_UNATTRIBUTED"].record(residual_ns, residual_ns)

    instrumented_cpu = non_io_cpu + residual_ns

    if DEBUG:
        print(f"\n  [DEBUG] session={session_id} task={task_id} turns={turns}")
        print(f"          wall={wall_ns_total/1e6:.1f}ms  process_cpu={thread_cpu_ns/1e6:.1f}ms")
        for cat in sorted(spans):
            s = spans[cat]
            print(f"          {cat:<28} cpu={s.cpu_ns/1e6:7.3f}ms  wall={s.wall_ns/1e6:9.2f}ms  n={s.count}")

    orch_cpu = spans["ORCH_SETUP"].cpu_ns + spans["ORCH_DISPATCH"].cpu_ns

    return {
        # Schema fields (Zachary-compatible)
        "session_id":                   session_id,
        "task_id":                      task_id,
        "turns":                        turns,
        "plan_len":                     task["plan_len"],
        "wall_s":                       wall_ns_total / 1e9,
        "thread_cpu_ns":                thread_cpu_ns,
        "process_cpu_ns":               thread_cpu_ns,
        "instrumented_cpu_ns":          instrumented_cpu,
        "reconcile_cpu_ns":             0,
        "orch_measured_cpu_ns":         orch_cpu,
        "orch_reconcile_cpu_ns":        0,
        "residual_unattributed_cpu_ns": residual_ns,
        "parallel_cpu_trim_ns":         0,
        "tool_call_counts":             tool_call_counts,
        "tool_call_sequence":           tool_call_sequence,
        "backend":                      BACKEND,
        "replay":                       replayed_call_count > 0,
        "model_call_count":             turns,
        "replayed_call_count":          replayed_call_count,
        "api_recorded_latency_ms_total": api_recorded_latency_ms_total,
        "api_replay_latency_ms_total":  api_replay_latency_ms_total,
        "provenance": {
            "measured":      instrumented_cpu - residual_ns,
            "step_inferred": 0,
            "residual":      residual_ns,
        },
        "search_locality": SEARCH_LOCALITY,
        # Internal fields — stripped before JSON output
        "_spans":   spans,
        "_wall_ns": wall_ns_total,
    }


# ---------------------------------------------------------------------------
# Seed runner
# ---------------------------------------------------------------------------

def _task_schedule(seed: int, n: int) -> list[str]:
    """Deterministically assign n tasks to a seed by cycling the task list."""
    ids    = list(TASKS.keys())
    offset = seed % len(ids)
    return [ids[(offset + i) % len(ids)] for i in range(n)]


def run_seed(backend: OpenAIChatBackend, seed: int) -> dict:
    """Run N_SESSIONS sessions for one seed. Returns a SeedArtifact-shaped dict."""
    task_ids = _task_schedule(seed, N_SESSIONS)
    sessions: list[dict] = []

    batch_t0 = wall_ns()

    for i, task_id in enumerate(task_ids):
        session_id = f"agent_{i}"
        print(f"  Seed {seed} | {i + 1}/{N_SESSIONS} | {task_id}  ({session_id})")
        try:
            sess = run_session(backend, session_id, task_id, TASKS[task_id])
            sessions.append(sess)
        except Exception as exc:
            print(f"    WARNING: {session_id} failed — {exc}")

    batch_wall_s = (wall_ns() - batch_t0) / 1e9

    # Aggregate per_category across all sessions in this seed
    seed_spans: dict[str, Span] = defaultdict(Span)
    for sess in sessions:
        for cat, span in sess["_spans"].items():
            seed_spans[cat].merge(span)

    thread_cpu_total    = sum(s["thread_cpu_ns"]       for s in sessions)
    instrumented_total  = sum(s["instrumented_cpu_ns"] for s in sessions)
    residual_total      = sum(s["residual_unattributed_cpu_ns"] for s in sessions)
    # On Windows, thread_cpu_total (process_time_ns, 15.6ms ticks) is far smaller
    # than the wall-proxy sum for TOOL_COMPUTE, making the old denominator wrong.
    # Use instrumented_total (wall-proxy sum incl. residual) as the denominator so
    # residual_fraction = "share of attributed CPU that is unattributed" — always in [0,1].
    residual_fraction   = residual_total / max(instrumented_total, 1)

    # per_session_category: {agent_N: {CATEGORY: CategoryMetrics}}
    per_session_category = {
        sess["session_id"]: {cat: span.to_dict() for cat, span in sess["_spans"].items()}
        for sess in sessions
    }

    # provenance_detail: flat keys "CATEGORY|agent_N|payload_profile|source"
    provenance_detail: dict[str, int] = {}
    for sess in sessions:
        agent = sess["session_id"]
        for cat, span in sess["_spans"].items():
            source = "residual" if cat == "RESIDUAL_UNATTRIBUTED" else "measured"
            provenance_detail[f"{cat}|{agent}|{PAYLOAD_PROFILE}|{source}"] = span.cpu_ns

    # per_task: aggregate category metrics per task ID
    task_sess_map: dict[str, list[dict]] = defaultdict(list)
    for sess in sessions:
        task_sess_map[sess["task_id"]].append(sess)

    per_task: dict[str, dict] = {}
    for task_id, task_sess in task_sess_map.items():
        task_spans: dict[str, Span] = defaultdict(Span)
        for s in task_sess:
            for cat, span in s["_spans"].items():
                task_spans[cat].merge(span)

        task_cpu       = sum(s["instrumented_cpu_ns"] for s in task_sess)
        amenable_strict = sum(task_spans[c].cpu_ns for c in HARNESS_STRICT_CATEGORIES if c in task_spans)
        amenable_broad  = sum(task_spans[c].cpu_ns for c in HARNESS_BROAD_CATEGORIES  if c in task_spans)

        per_task[task_id] = {
            "categories":          {c: s.to_dict() for c, s in task_spans.items()},
            "sessions":            [s["session_id"] for s in task_sess],
            "instrumented_cpu_ns": task_cpu,
            "amenable_strict_ns":  amenable_strict,
            "amenable_broad_ns":   amenable_broad,
            "amenable_strict_share": amenable_strict / max(task_cpu, 1),
            "amenable_broad_share":  amenable_broad  / max(task_cpu, 1),
        }

    # per_task_wall_cpu: one entry per task (last session for that task)
    per_task_wall_cpu: dict[str, dict] = {}
    for task_id, task_sess in task_sess_map.items():
        s        = task_sess[-1]
        cats     = s["_spans"]
        wall_ns  = s["_wall_ns"]
        host_ms  = s["instrumented_cpu_ns"] / 1e6
        tool_ms  = cats["TOOL_COMPUTE"].cpu_ns / 1e6 if "TOOL_COMPUTE" in cats else 0.0
        harness_ms = sum(cats[c].cpu_ns for c in HARNESS_STRICT_CATEGORIES if c in cats) / 1e6
        llm_wall_ns  = cats["HTTP_CLIENT"].wall_ns if "HTTP_CLIENT" in cats else 0
        tool_wall_ns = cats["TOOL_COMPUTE"].wall_ns if "TOOL_COMPUTE" in cats else 0

        per_task_wall_cpu[task_id] = {
            "session_id":                         s["session_id"],
            "session_wall_s":                     wall_ns / 1e9,
            "llm_io_wait_s":                      llm_wall_ns / 1e9,
            "remote_tool_io_wait_s":              tool_wall_ns / 1e9,
            "non_llm_wall_s":                     max(0, wall_ns - llm_wall_ns) / 1e9,
            "host_cpu_ms":                        host_ms,
            "llm_io_pct_of_session_wall":         100.0 * llm_wall_ns  / max(wall_ns, 1),
            "remote_tool_io_pct_of_session_wall": 100.0 * tool_wall_ns / max(wall_ns, 1),
            "host_cpu_pct_of_session_wall":       100.0 * s["instrumented_cpu_ns"] / max(wall_ns, 1),
            "tool_cpu_ms":                        tool_ms,
            "harness_cpu_ms":                     harness_ms,
            "cpu_by_category_ms":                 {cat: span.cpu_ns / 1e6 for cat, span in cats.items()},
            "tool_call_counts":                   s["tool_call_counts"],
        }

    # Strip internal _spans / _wall_ns before writing to output
    clean_sessions = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in sessions
    ]

    return {
        "experiment":      "real_agent_breakdown",
        "result_validity": "valid",    # overwritten after cross-seed check in main()
        "generated_utc":   datetime.now(timezone.utc).isoformat(),
        "setup_ref":       _get_setup_ref(),
        "config": {
            "profile":          PROFILE,
            "payload_profile":  PAYLOAD_PROFILE,
            "search_locality":  SEARCH_LOCALITY,
            "seed":             seed,
            "sessions":         N_SESSIONS,
            "mode":             f"sync/{BACKEND}",
            "llm_median_scale": 1.0,
            "instr_version":    INSTR_VERSION,
        },
        "batch_wall_s": batch_wall_s,
        "run": {
            "env":    {},
            "config": {},
            "per_category":          {c: s.to_dict() for c, s in seed_spans.items()},
            "per_session":           clean_sessions,
            "per_session_category":  per_session_category,
            "residual_cpu_ns":       residual_total,
            "residual_fraction":     residual_fraction,
            "totals": {
                "thread_cpu_ns":       thread_cpu_total,
                "wall_ns":             int(batch_wall_s * 1e9),
                "instrumented_cpu_ns": instrumented_total,
            },
            "os_times_user_sys": {},
            "timeline":          [],
            "provenance_totals": {
                "measured":      instrumented_total - residual_total,
                "step_inferred": 0,
                "residual":      residual_total,
            },
            "provenance_detail": provenance_detail,
        },
        "per_task":          per_task,
        "per_task_wall_cpu": per_task_wall_cpu,
        # Internal: retained for build_aggregate cross-seed access, stripped in main()
        "_sessions": sessions,
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    """Compute Stats shape: median, q1, q3, iqr, n, min, max."""
    if not values:
        return {"median": 0.0, "q1": 0.0, "q3": 0.0, "iqr": 0.0, "n": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = p / 100 * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    q1, med, q3 = pct(25), pct(50), pct(75)
    return {"median": med, "q1": q1, "q3": q3, "iqr": q3 - q1, "n": float(n), "min": s[0], "max": s[-1]}


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------

def build_aggregate(artifacts: list[dict]) -> dict:
    """Compute cross-seed aggregate statistics. Mirrors Zachary's aggregate object."""
    seeds = [a["config"]["seed"] for a in artifacts]

    def pooled_pct(cats: list[str]) -> list[float]:
        out = []
        for a in artifacts:
            total   = a["run"]["totals"]["instrumented_cpu_ns"]
            cat_cpu = sum(a["run"]["per_category"].get(c, {}).get("cpu_ns", 0) for c in cats)
            out.append(100.0 * cat_cpu / max(total, 1))
        return out

    orch_cats = ["ORCH_SETUP", "ORCH_DISPATCH"]
    zero_dist = [0.0] * len(artifacts)

    task_samples: dict[str, list[float]] = defaultdict(list)
    for a in artifacts:
        for task_id, td in a["per_task"].items():
            task_samples[task_id].append(td["instrumented_cpu_ns"] / 1e6)

    batch_cpu_ms = [a["run"]["totals"]["instrumented_cpu_ns"] / 1e6 for a in artifacts]

    return {
        "n_seeds":                       len(artifacts),
        "seeds":                         seeds,
        "harness_strict_definition":     HARNESS_STRICT_DEFINITION,
        "harness_broad_definition":      HARNESS_BROAD_DEFINITION,
        "attribution_doc":               ATTRIBUTION_DOC,
        "comparison_type":               "distribution_over_seeds",
        "batch_host_cpu_ms":             _stats(batch_cpu_ms),
        "pooled_tool_compute_pct":       _stats(pooled_pct(["TOOL_COMPUTE"])),
        "pooled_orch_pct":               _stats(pooled_pct(orch_cats)),
        "pooled_orch_measured_pct":      _stats(pooled_pct(orch_cats)),
        "pooled_orch_reconcile_pct":     _stats(zero_dist),
        "pooled_harness_apu_pct":        _stats(pooled_pct(list(HARNESS_STRICT_CATEGORIES))),
        "pooled_harness_strict_pct":     _stats(pooled_pct(list(HARNESS_STRICT_CATEGORIES))),
        "pooled_harness_broad_pct":      _stats(pooled_pct(list(HARNESS_BROAD_CATEGORIES))),
        "pooled_residual_unattributed_pct": _stats(pooled_pct(["RESIDUAL_UNATTRIBUTED"])),
        "pooled_measured_pct": _stats([
            a["run"]["provenance_totals"]["measured"] /
            max(a["run"]["totals"]["instrumented_cpu_ns"], 1) * 100
            for a in artifacts
        ]),
        "pooled_step_inferred_pct":      _stats(zero_dist),
        "pooled_residual_provenance_pct": _stats([
            a["run"]["provenance_totals"]["residual"] /
            max(a["run"]["totals"]["instrumented_cpu_ns"], 1) * 100
            for a in artifacts
        ]),
        "pooled_client_http_pct":        _stats(pooled_pct(["CLIENT_HTTP"])),
        "pooled_client_parse_pct":       _stats(pooled_pct(["CLIENT_PARSE"])),
        "pooled_event_loop_pct":         _stats(zero_dist),
        "pooled_framework_pct":          _stats(pooled_pct(["FRAMEWORK"])),
        "pooled_threadpool_pct":         _stats(zero_dist),
        "orch_reconcile_share_of_orch_pct": _stats(zero_dist),
        "per_task_host_cpu_ms": {
            tid: _stats(samps) for tid, samps in task_samples.items()
        },
    }


# ---------------------------------------------------------------------------
# Result validity gate
# ---------------------------------------------------------------------------

def _check_validity(artifacts: list[dict]) -> str:
    """Return 'publishable' when all seeds complete and median residual < 15%.

    Windows process_time_ns (15.6ms ticks) creates per-seed outliers when a seed
    happens to land on more tick boundaries. We gate on the median across seeds
    rather than any single seed to tolerate these sampling artifacts.
    """
    if len(artifacts) < N_SEEDS:
        return f"insufficient_seeds_{len(artifacts)}_of_{N_SEEDS}"
    fracs = sorted(a["run"]["residual_fraction"] for a in artifacts)
    median_frac = fracs[len(fracs) // 2]
    if median_frac > 0.15:
        return f"residual_too_high_median_{median_frac:.3f}"
    return "publishable"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_env() -> dict:
    try:
        import psutil
        ram_gb         = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        cores_physical = psutil.cpu_count(logical=False) or 1
        cores_logical  = psutil.cpu_count(logical=True)  or 1
    except ImportError:
        ram_gb         = 0.0
        cores_physical = os.cpu_count() or 1
        cores_logical  = os.cpu_count() or 1

    return {
        "python":     platform.python_version(),
        "platform":   platform.platform(),
        "cpu_model":  platform.processor() or "unknown",
        "blas_pin": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS":      os.environ.get("MKL_NUM_THREADS",      "unset"),
            "OMP_NUM_THREADS":      os.environ.get("OMP_NUM_THREADS",      "unset"),
        },
        "cores_logical":  cores_logical,
        "cores_physical": cores_physical,
        "ram_gb":         ram_gb,
    }


def _get_git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty_raw = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty       = "yes" if dirty_raw else "no"
        dirty_paths = [ln.split()[-1] for ln in dirty_raw.splitlines() if ln] if dirty_raw else []
    except Exception:
        commit, dirty, dirty_paths = "unknown", "unknown", []
    return {"commit": commit, "dirty": dirty, "dirty_paths": dirty_paths}


def _get_setup_ref() -> dict:
    digest_src = json.dumps(
        {"model": MODEL, "instr_version": INSTR_VERSION, "tasks": list(TASKS.keys())},
        sort_keys=True,
    )
    suite_src = json.dumps(sorted(TASKS.keys()))
    return {
        "setup_digest":      hashlib.sha256(digest_src.encode()).hexdigest()[:16],
        "task_suite_digest": hashlib.sha256(suite_src.encode()).hexdigest()[:16],
        "cpu_model":         platform.processor() or "unknown",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    traces_root = Path(TRACES_ROOT_ENV) if TRACES_ROOT_ENV else None
    replay_cache = ReplayCache(mode=REPLAY_MODE, traces_root=traces_root)
    backend = OpenAIChatBackend(client=client, replay_cache=replay_cache)

    git_info  = _get_git_info()
    env_info  = _get_env()
    setup_ref = _get_setup_ref()
    seeds     = list(range(N_SEEDS))

    raw_artifacts: list[dict] = []
    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Seed {seed}")
        print(f"{'=' * 60}")
        artifact = run_seed(backend, seed)
        raw_artifacts.append(artifact)

    result_validity = _check_validity(raw_artifacts)

    # Strip internal fields and stamp final validity on every seed artifact
    clean_artifacts = []
    for a in raw_artifacts:
        ca = {k: v for k, v in a.items() if not k.startswith("_")}
        ca["result_validity"] = result_validity
        clean_artifacts.append(ca)

    aggregate = build_aggregate(raw_artifacts)   # uses _sessions on raw artifacts

    output = {
        "experiment":      "replication_batch",
        "result_validity": result_validity,
        "generated_utc":   datetime.now(timezone.utc).isoformat(),
        "setup_ref":       setup_ref,
        "git":             git_info,
        "measurement_git": git_info,
        "env":             env_info,
        "config": {
            "profile":          PROFILE,
            "search_locality":  SEARCH_LOCALITY,
            "seeds":            seeds,
            "sessions":         N_SESSIONS,
            "backend":          BACKEND,
            "comparison_type":  "distribution_over_seeds",
            "allow_dirty":      False,
            "instr_version":    INSTR_VERSION,
        },
        "aggregate":          aggregate,
        "per_seed_artifacts": clean_artifacts,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    med = aggregate["batch_host_cpu_ms"]["median"]
    print(f"\n{'=' * 60}")
    print(f"Output written to  : {OUTPUT_PATH}")
    print(f"result_validity    : {result_validity}")
    print(f"seeds completed    : {len(clean_artifacts)}")
    print(f"batch_host_cpu_ms  : {med:.1f} ms  (median across seeds)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
