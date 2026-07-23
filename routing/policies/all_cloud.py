"""Route every task to the cloud backend."""

from __future__ import annotations

from typing import Any

from harness.backends.base import Backend
from routing.policies.base import RoutingPolicy


class AllCloudPolicy(RoutingPolicy):
    """Always routes to cloud."""

    def __init__(self, cloud_backend: Backend) -> None:
        self.cloud_backend = cloud_backend

    def route(self, task_id: str, category: str, step_context: dict[str, Any]) -> Backend:
        return self.cloud_backend
