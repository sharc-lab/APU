"""Deterministic tool environment for the frozen BFCL-style workload.

The world is data loaded from a task-list JSON file, not a live service. That is a
measurement decision, not a convenience: this slice measures **latency-driven routing**, and a
network-backed tool would inject wall-clock variance indistinguishable from the effect under test.
Every tool here returns in microseconds and is exactly reproducible.

Tool execution is identical across conditions. Only the model deciding *which* tool to call
changes.

C2 adds ``retrieve_documents`` and optional large corpus payloads. Worlds without a corpus keep
the Stage-1 tiny-string behaviour (backward compatible with ``bfcl_slice_v1.json``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from seam.agent.document_corpus import DocumentCorpus
from seam.backends.base import ToolSpec

__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_SPECS",
    "TaskSpec",
    "ToolWorld",
    "Workload",
    "load_workload",
    "normalize_answer",
]

#: Arithmetic only. ``eval`` on model output would be a remote-code-execution hole, and a model
#: that emits ``__import__`` is a finding to log rather than a command to run.
_SAFE_EXPR_RE: Final = re.compile(r"^[0-9+\-*/(). ]+$")


# ==================================================================================================
# Tool definitions - identical for every backend and every condition
# ==================================================================================================

TOOL_SPECS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="list_files",
        description="List the file names in a directory.",
        input_schema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Absolute directory path, for example /docs",
                }
            },
            "required": ["directory"],
        },
    ),
    ToolSpec(
        name="read_file",
        description="Read the full text contents of a file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path, for example /docs/README.txt",
                }
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="retrieve_documents",
        description=(
            "Retrieve 1-3 research document chunks matching a query. Each chunk is long "
            "(hundreds to thousands of tokens). Use this for multi-document research before "
            "submit_answer."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, for example 'budget Q1 Northgate'",
                }
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="lookup_employee",
        description="Look up an employee record by full name. Returns role, city, and id.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full employee name"},
            },
            "required": ["name"],
        },
    ),
    ToolSpec(
        name="get_weather",
        description="Get the current weather for a city. Returns temperature in Celsius "
        "and a condition word.",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression. Supports + - * / and parentheses.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, for example (1200 + 1850) / 2",
                }
            },
            "required": ["expression"],
        },
    ),
    ToolSpec(
        name="submit_answer",
        description="Submit the final answer. Call this exactly once, when you have the answer.",
        input_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The final answer, and nothing else"},
            },
            "required": ["answer"],
        },
    ),
)

#: The one system prompt. Identical for every model and every target - see spec §7 M3.1.
SYSTEM_PROMPT: Final = (
    "You are a precise assistant with access to tools. Use the tools to find the information you "
    "need, then call submit_answer exactly once with the final answer and nothing else. Do not "
    "guess: if you need a fact, call a tool to get it. For research questions, call "
    "retrieve_documents and/or read_file on corpus documents before submit_answer - answers are "
    "not available from a single short string. Keep the answer short - just the value asked for, "
    "with no units, punctuation, or explanation."
)


# ==================================================================================================
# World
# ==================================================================================================


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    prompt: str
    expected: str
    min_tool_calls: int


@dataclass(frozen=True, slots=True)
class Workload:
    benchmark: str
    world: ToolWorld
    tasks: tuple[TaskSpec, ...]
    sha256: str
    path: Path


class ToolWorld:
    """Executes tool calls against the frozen world."""

    def __init__(self, world: dict[str, Any]) -> None:
        self._directories: dict[str, list[str]] = {
            str(k): list(v) for k, v in (world.get("directories") or {}).items()
        }
        self._files: dict[str, str] = {
            str(k): str(v) for k, v in (world.get("files") or {}).items()
        }
        self._employees: dict[str, dict[str, Any]] = world.get("employees", {})
        self._weather: dict[str, dict[str, Any]] = world.get("weather", {})
        self._corpus: DocumentCorpus | None = None
        corpus_seed = world.get("corpus_seed")
        if corpus_seed is not None:
            planted = world.get("planted_facts") or {}
            self._corpus = DocumentCorpus(
                seed=int(corpus_seed),
                n_chunks=int(world.get("corpus_n_chunks", 48)),
                planted_facts={str(k): dict(v) for k, v in planted.items()},
            )
            overlay = self._corpus.world_overlay()
            for directory, entries in overlay["directories"].items():
                merged = {
                    (entry.rsplit("/", 1)[-1] if "/" in entry else entry)
                    for entry in list(self._directories.get(directory, [])) + list(entries)
                }
                self._directories[directory] = sorted(merged)
            for path, content in overlay["files"].items():
                self._files.setdefault(path, content)

    @property
    def corpus(self) -> DocumentCorpus | None:
        return self._corpus

    def begin_task(self) -> None:
        """Reset per-task retrieve call indexing so trajectories are independent."""
        if self._corpus is not None:
            self._corpus.reset_call_index()

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, str | None]:
        """Run one tool call.

        Returns ``(result_text, error)``. An unknown tool or a bad argument produces an error
        *message returned to the model*, not an exception: recovering from a bad call is part of
        the behavior under study, and crashing the run would discard that observation.
        """
        handler = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "retrieve_documents": self._retrieve_documents,
            "lookup_employee": self._lookup_employee,
            "get_weather": self._get_weather,
            "calculator": self._calculator,
        }.get(name)
        if handler is None:
            return "", f"unknown tool {name!r}"
        try:
            return handler(arguments), None
        except KeyError as exc:
            return "", f"missing required argument {exc}"

    def _list_files(self, args: dict[str, Any]) -> str:
        directory = str(args["directory"]).rstrip("/") or "/"
        entries = self._directories.get(directory)
        if entries is None:
            return f"error: no such directory {directory}"
        # Normalize to basenames for listing when corpus paths are stored absolute.
        names = []
        for entry in entries:
            names.append(entry.rsplit("/", 1)[-1] if "/" in entry else entry)
        return json.dumps(sorted(set(names)))

    def _read_file(self, args: dict[str, Any]) -> str:
        path = str(args["path"])
        if self._corpus is not None:
            chunk = self._corpus.get_by_path(path)
            if chunk is not None:
                return chunk.text
        content = self._files.get(path)
        if content is None:
            return f"error: no such file {path}"
        return content

    def _retrieve_documents(self, args: dict[str, Any]) -> str:
        if self._corpus is None:
            return "error: retrieve_documents is unavailable in this world (no corpus)"
        query = str(args["query"])
        chunks = self._corpus.retrieve(query)
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "path": chunk.path,
                "topic": chunk.topic,
                "token_proxy": chunk.token_proxy,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        return json.dumps(payload)

    def _lookup_employee(self, args: dict[str, Any]) -> str:
        name = str(args["name"]).strip()
        record = self._employees.get(name)
        if record is None:
            return f"error: no employee named {name}"
        return json.dumps(record)

    def _get_weather(self, args: dict[str, Any]) -> str:
        city = str(args["city"]).strip()
        record = self._weather.get(city)
        if record is None:
            return f"error: no weather data for {city}"
        return json.dumps(record)

    def _calculator(self, args: dict[str, Any]) -> str:
        expression = str(args["expression"]).strip()
        if not _SAFE_EXPR_RE.match(expression):
            return "error: expression may contain only digits, + - * / ( ) and spaces"
        try:
            # Gated by _SAFE_EXPR_RE above: digits and arithmetic operators only, no names.
            value = eval(expression, {"__builtins__": {}}, {})
        except (SyntaxError, ZeroDivisionError, TypeError) as exc:
            return f"error: {type(exc).__name__}"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)


def load_workload(path: Path) -> Workload:
    """Load and hash the frozen task list.

    The hash is returned so the caller can assert it against the value pinned in
    the experiment config. A silently edited task list is the easiest way to invalidate a
    comparison without noticing.
    """
    from seam.hashing import sha256_file

    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = tuple(
        TaskSpec(
            task_id=str(t["task_id"]),
            prompt=str(t["prompt"]),
            expected=str(t["expected"]),
            min_tool_calls=int(t.get("min_tool_calls", 1)),
        )
        for t in raw["tasks"]
    )
    return Workload(
        benchmark=str(raw["benchmark"]),
        world=ToolWorld(raw["world"]),
        tasks=tasks,
        sha256=sha256_file(path),
        path=path,
    )


def normalize_answer(value: str) -> str:
    """Canonicalize an answer for exact machine checking.

    Deliberately narrow: case, surrounding whitespace, trailing punctuation, thousands separators,
    and a trailing ``.0`` are normalized away. Nothing semantic is. A grader that is generous about
    meaning would let the local model's failures be scored as successes, which is precisely the
    quantity the unconstrained success-rate check needs to be honest about.
    """
    text = value.strip().strip(".,;:!?\"'").replace(",", "")
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text.casefold()
    if math.isfinite(number) and number == int(number):
        return str(int(number))
    return str(number)
