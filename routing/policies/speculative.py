"""Speculative dual-execution policy with agreement-based cloud charging."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

from harness.backends.base import Backend, ModelCallResult
from routing.policies.base import RoutingPolicy


class SpeculativePolicy(RoutingPolicy):
    """Run local+cloud, commit local on agreement, rollback to cloud on disagreement."""

    requires_speculative_dual = True

    def __init__(
        self,
        *,
        cloud_backend: Backend,
        local_backend: Backend,
        text_similarity_threshold: float = 0.88,
    ) -> None:
        self.cloud_backend = cloud_backend
        self.local_backend = local_backend
        self.text_similarity_threshold = float(text_similarity_threshold)
        self._agreement_memory: set[tuple[str, str]] = set()

    def route(self, task_id: str, category: str, step_context: dict[str, Any]) -> Backend:
        # Selection happens after dual execution; placeholder for interface compliance.
        return self.local_backend

    @staticmethod
    def _extract_text(result: ModelCallResult) -> str:
        choices = result.response_json.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _tool_calls(result: ModelCallResult) -> list[dict[str, Any]]:
        choices = result.response_json.get("choices", [])
        if not choices:
            return []
        message = choices[0].get("message", {})
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            return calls
        return []

    @staticmethod
    def _norm_tool_signature(calls: list[dict[str, Any]]) -> str:
        normalized = []
        for call in calls:
            function = call.get("function") or {}
            normalized.append(
                {
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                }
            )
        return json.dumps(normalized, sort_keys=True)

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        toks_a = re.findall(r"[a-z0-9_]+", a.lower())
        toks_b = re.findall(r"[a-z0-9_]+", b.lower())
        if not toks_a and not toks_b:
            return 1.0
        if not toks_a or not toks_b:
            return 0.0
        va = Counter(toks_a)
        vb = Counter(toks_b)
        keys = set(va) | set(vb)
        dot = sum(va[k] * vb[k] for k in keys)
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _step_signature(category: str, task_id: str, prompt: str) -> str:
        payload = json.dumps({"category": category, "task_id": task_id, "prompt": prompt}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def decide(
        self,
        *,
        task_id: str,
        category: str,
        prompt: str,
        local_result: ModelCallResult,
        cloud_result: ModelCallResult,
    ) -> dict[str, Any]:
        """Compare speculative outputs and decide commit/rollback and charging."""
        step_sig = self._step_signature(category, task_id, prompt)

        local_calls = self._tool_calls(local_result)
        cloud_calls = self._tool_calls(cloud_result)
        exact_tool_match = False
        similarity = None

        if local_calls or cloud_calls:
            exact_tool_match = self._norm_tool_signature(local_calls) == self._norm_tool_signature(cloud_calls)
            agreed = exact_tool_match
            comparator = "tool_call_exact_match"
        else:
            similarity = self._text_similarity(self._extract_text(local_result), self._extract_text(cloud_result))
            agreed = similarity >= self.text_similarity_threshold
            comparator = "text_similarity"

        previously_agreed = (category, step_sig) in self._agreement_memory
        if agreed:
            self._agreement_memory.add((category, step_sig))
        charge_cloud_tokens = not (agreed and previously_agreed)

        chosen_backend = self.local_backend if agreed else self.cloud_backend
        committed_result = local_result if agreed else cloud_result

        return {
            "agreed": agreed,
            "rollback": not agreed,
            "comparator": comparator,
            "exact_tool_call_match": exact_tool_match,
            "text_similarity": similarity,
            "threshold": self.text_similarity_threshold,
            "step_signature": step_sig,
            "charge_cloud_tokens": charge_cloud_tokens,
            "known_agreement": previously_agreed,
            "chosen_backend": chosen_backend,
            "committed_result": committed_result,
            "local_result": local_result,
            "cloud_result": cloud_result,
        }
