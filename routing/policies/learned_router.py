"""Learned router policy driven by a distilled classifier artifact."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from harness.backends.base import Backend
from routing.policies.base import RoutingPolicy


class LearnedRouterPolicy(RoutingPolicy):
    """Route steps to local/cloud using a trained adequacy classifier."""

    def __init__(
        self,
        *,
        cloud_backend: Backend,
        local_backend: Backend,
        model_path: str | Path,
        threshold: float = 0.5,
    ) -> None:
        self.cloud_backend = cloud_backend
        self.local_backend = local_backend
        self.threshold = float(threshold)

        model_path = Path(model_path)
        payload = pickle.loads(model_path.read_bytes())
        self.model = payload["best_model"]
        self.feature_names = list(payload.get("feature_names", []))

    def _vectorize(self, step_context: dict[str, Any]) -> list[float]:
        values = []
        for name in self.feature_names:
            raw = step_context.get(name, 0.0)
            try:
                values.append(float(raw))
            except Exception:
                values.append(0.0)
        return values

    def route(self, task_id: str, category: str, step_context: dict[str, Any]) -> Backend:
        x = [self._vectorize(step_context)]
        if hasattr(self.model, "predict_proba"):
            p_local = float(self.model.predict_proba(x)[0][1])
        else:
            pred = int(self.model.predict(x)[0])
            p_local = float(pred)

        step_context["learned_router_p_local_adequate"] = p_local
        step_context["learned_router_threshold"] = self.threshold
        return self.local_backend if p_local >= self.threshold else self.cloud_backend
