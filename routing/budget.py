"""Budget tracking and cloud-to-local fallback enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.backends.base import Backend


@dataclass
class BudgetTracker:
    """Tracks cloud token usage and enforces a hard cap."""

    cloud_token_cap: int
    cloud_input_tokens: int = 0
    cloud_output_tokens: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cloud_total_tokens(self) -> int:
        return self.cloud_input_tokens + self.cloud_output_tokens

    @property
    def remaining_cloud_tokens(self) -> int:
        return max(0, self.cloud_token_cap - self.cloud_total_tokens)

    def enforce(
        self,
        *,
        task_id: str,
        category: str,
        requested_backend: Backend,
        local_backend: Backend,
        step_context: dict[str, Any] | None = None,
    ) -> Backend:
        """Force local fallback when cloud budget is exhausted."""
        step_context = step_context or {}
        forced_local = requested_backend.is_cloud and self.remaining_cloud_tokens <= 0
        selected_backend = local_backend if forced_local else requested_backend

        self.decisions.append(
            {
                "task_id": task_id,
                "category": category,
                "requested_backend": requested_backend.name,
                "selected_backend": selected_backend.name,
                "forced_local": forced_local,
                "reason": "cloud_budget_exhausted" if forced_local else "policy_selected",
                "cloud_input_tokens": self.cloud_input_tokens,
                "cloud_output_tokens": self.cloud_output_tokens,
                "cloud_total_tokens": self.cloud_total_tokens,
                "cloud_token_cap": self.cloud_token_cap,
                "remaining_cloud_tokens": self.remaining_cloud_tokens,
                "step_context": step_context,
            }
        )
        return selected_backend

    def record_usage(self, backend: Backend, token_counts: dict[str, int]) -> None:
        """Accumulate token usage after a model call."""
        if not backend.is_cloud:
            return
        self.cloud_input_tokens += int(token_counts.get("prompt_tokens", 0))
        self.cloud_output_tokens += int(token_counts.get("completion_tokens", 0))

    def to_artifact(self) -> dict[str, Any]:
        """Export budget and routing decisions for run artifacts."""
        return {
            "cloud_input_tokens": self.cloud_input_tokens,
            "cloud_output_tokens": self.cloud_output_tokens,
            "cloud_total_tokens": self.cloud_total_tokens,
            "cloud_token_cap": self.cloud_token_cap,
            "remaining_cloud_tokens": self.remaining_cloud_tokens,
            "routing_decisions": self.decisions,
        }
