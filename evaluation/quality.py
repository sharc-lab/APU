"""Per-task quality scoring for APU benchmark runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harness.backends.cloud_openai import CloudOpenAIBackend
from harness.replay import ReplayMode


DETERMINISTIC_SCORERS = {
    "CN-01": "compute_numerical",
}


def _extract_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _score_cn01(text: str) -> float:
    """Programmatic check for CN-01 expected values."""
    hits = 0
    if "338350" in text:
        hits += 1
    if "325" in text:
        hits += 1
    # Explicit formula reference increases confidence.
    if "n*(n+1)*(2*n+1)/6" in text or "2*n+1" in text:
        hits += 1
    return round((hits / 3) * 10.0, 2)


class QualityEvaluator:
    """Scores deterministic tasks directly and open-ended tasks via cached LLM judge."""

    def __init__(
        self,
        *,
        judge_model: str = "gpt-4o-mini",
        judge_prompt_version: str = "rubric_v1",
        replay_mode: ReplayMode | str = ReplayMode.AUTO,
    ) -> None:
        self.judge_model = judge_model
        self.judge_prompt_version = judge_prompt_version
        self.prompt_text = self._load_prompt(judge_prompt_version)
        self.judge_backend = CloudOpenAIBackend(model=judge_model, replay_mode=replay_mode)

    @staticmethod
    def _load_prompt(version: str) -> str:
        prompt_path = Path(__file__).parent / "judge_prompts" / f"{version}.md"
        return prompt_path.read_text(encoding="utf-8")

    def score_task(
        self,
        *,
        task_id: str,
        task_prompt: str,
        response_json: dict[str, Any],
    ) -> dict[str, Any]:
        """Return score metadata with deterministic or judge-backed method."""
        output_text = _extract_text(response_json)

        if DETERMINISTIC_SCORERS.get(task_id) == "compute_numerical":
            score = _score_cn01(output_text)
            return {
                "score": score,
                "method": "deterministic_programmatic",
                "judge_model": None,
                "judge_prompt_version": None,
            }

        judge_messages = [
            {"role": "system", "content": self.prompt_text},
            {
                "role": "user",
                "content": (
                    "Task ID: " + task_id + "\n\n"
                    "Task prompt:\n" + task_prompt + "\n\n"
                    "Model output:\n" + output_text + "\n\n"
                    "Provide only JSON with score and rationale."
                ),
            },
        ]

        # Judge calls are routed through ReplayCache via CloudOpenAIBackend.
        judge_result = self.judge_backend.model_call(
            messages=judge_messages,
            tools=None,
            temperature=0.0,
            seed=0,
            model=self.judge_model,
        )
        judge_text = _extract_text(judge_result.response_json)
        score = self._parse_score(judge_text)

        return {
            "score": score,
            "method": "llm_judge",
            "judge_model": self.judge_model,
            "judge_prompt_version": self.judge_prompt_version,
            "judge_replayed": judge_result.replayed,
            "judge_cache_key": judge_result.cache_key,
        }

    @staticmethod
    def _parse_score(judge_text: str) -> float:
        try:
            payload = json.loads(judge_text)
            value = float(payload.get("score", 0.0))
            return max(0.0, min(10.0, value))
        except Exception:
            pass

        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", judge_text)
        if not match:
            return 0.0
        value = float(match.group(1))
        return max(0.0, min(10.0, value))
