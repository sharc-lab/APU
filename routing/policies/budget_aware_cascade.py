"""Budget-aware cascade policy with dynamic escalation threshold."""

from __future__ import annotations

from typing import Any

from routing.policies.cascade import CascadePolicy


class BudgetAwareCascadePolicy(CascadePolicy):
    """Cascade policy where escalation threshold depends on remaining budget."""

    def __init__(
        self,
        *,
        cloud_backend,
        local_backend,
        theta_min: float = 0.35,
        theta_max: float = 0.80,
        **kwargs,
    ) -> None:
        super().__init__(
            cloud_backend=cloud_backend,
            local_backend=local_backend,
            escalation_threshold=theta_min,
            **kwargs,
        )
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

    def _theta(self, step_context: dict[str, Any]) -> float:
        b = float(step_context.get("remaining_budget_fraction", 1.0))
        b = min(1.0, max(0.0, b))
        theta = self.theta_min + (self.theta_max - self.theta_min) * (1.0 - b)
        return min(1.0, max(0.0, theta))
