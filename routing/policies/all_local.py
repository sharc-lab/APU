"""Route every task to the local backend."""

from __future__ import annotations

from typing import Any

from harness.backends.base import Backend
from routing.policies.base import RoutingPolicy


class AllLocalPolicy(RoutingPolicy):
    """Always routes to local."""

    def __init__(self, local_backend: Backend) -> None:
        self.local_backend = local_backend

    def route(self, task_id: str, category: str, step_context: dict[str, Any]) -> Backend:
        return self.local_backend
