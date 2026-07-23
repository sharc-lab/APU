"""Cascade routing policy with heuristic confidence-based escalation."""

from __future__ import annotations

from typing import Any

from harness.backends.base import Backend
from routing.policies.base import RoutingPolicy


class CascadePolicy(RoutingPolicy):
    """Attempt local first; escalate to cloud when local confidence is low."""

    requires_local_attempt = True

    def __init__(
        self,
        *,
        cloud_backend: Backend,
        local_backend: Backend,
        escalation_threshold: float = 0.55,
        min_output_chars: int = 32,
        max_output_chars: int = 4000,
        uncertainty_tokens: tuple[str, ...] = (
            "not sure",
            "uncertain",
            "i might be wrong",
            "cannot determine",
            "insufficient information",
        ),
    ) -> None:
        self.cloud_backend = cloud_backend
        self.local_backend = local_backend
        self.escalation_threshold = float(escalation_threshold)
        self.min_output_chars = int(min_output_chars)
        self.max_output_chars = int(max_output_chars)
        self.uncertainty_tokens = tuple(t.lower() for t in uncertainty_tokens)

    def _theta(self, step_context: dict[str, Any]) -> float:
        return self.escalation_threshold

    def _confidence(self, step_context: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        output_text = str(step_context.get("local_output_text", ""))
        output_len = int(step_context.get("local_output_len", len(output_text)))
        malformed_tool_call = bool(step_context.get("malformed_tool_call", False))
        retry_count = int(step_context.get("retry_count", 0))

        text_lower = output_text.lower()
        has_uncertainty = any(tok in text_lower for tok in self.uncertainty_tokens)
        len_anomaly = output_len < self.min_output_chars or output_len > self.max_output_chars

        risk = 0.0
        if len_anomaly:
            risk += 0.35
        if malformed_tool_call:
            risk += 0.35
        if has_uncertainty:
            risk += 0.20
        if retry_count > 0:
            risk += min(0.10, retry_count * 0.05)

        risk = min(1.0, risk)
        confidence = max(0.0, 1.0 - risk)
        details = {
            "len_anomaly": len_anomaly,
            "malformed_tool_call": malformed_tool_call,
            "has_uncertainty_token": has_uncertainty,
            "retry_count": retry_count,
            "risk": risk,
        }
        return confidence, details

    def route(self, task_id: str, category: str, step_context: dict[str, Any]) -> Backend:
        theta = self._theta(step_context)
        confidence, details = self._confidence(step_context)
        escalate = confidence < theta

        step_context["theta"] = theta
        step_context["confidence"] = confidence
        step_context["cascade_details"] = details
        step_context["escalate"] = escalate

        return self.cloud_backend if escalate else self.local_backend
