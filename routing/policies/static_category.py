"""Category-to-backend static router loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from harness.backends.base import Backend
from routing.policies.base import RoutingPolicy


class StaticCategoryPolicy(RoutingPolicy):
    """Routes by category using an editable YAML mapping table."""

    def __init__(
        self,
        *,
        cloud_backend: Backend,
        local_backend: Backend,
        mapping_path: str | Path,
        default_backend: str = "cloud",
    ) -> None:
        self.cloud_backend = cloud_backend
        self.local_backend = local_backend
        self.mapping_path = Path(mapping_path)
        self.default_backend = default_backend
        self._mapping = self._load_mapping(self.mapping_path)

    def route(self, task_id: str, category: str, step_context: dict[str, Any]) -> Backend:
        label = self._mapping.get(category, self.default_backend)
        if label == "local":
            return self.local_backend
        return self.cloud_backend

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, str]:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        table = data.get("category_to_backend", {})
        out: dict[str, str] = {}
        for category, backend in table.items():
            backend_label = str(backend).strip().lower()
            if backend_label not in {"cloud", "local"}:
                raise ValueError(f"Unsupported backend label '{backend}' for category '{category}'")
            out[str(category)] = backend_label
        return out
