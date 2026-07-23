"""Routing policy interface for backend selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from harness.backends.base import Backend


class RoutingPolicy(ABC):
    """Policy abstraction that maps task context to a backend."""

    @abstractmethod
    def route(self, task_id: str, category: str, step_context: dict[str, Any]) -> Backend:
        """Pick a backend for the current step."""
        raise NotImplementedError
