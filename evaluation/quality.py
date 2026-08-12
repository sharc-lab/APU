"""Per-task quality scoring for APU benchmark runs.

Scoring hierarchy (applied in order, first hit wins):
  1. Deterministic programmatic: hard-coded scorer for CN-01.
  2. LLM judge: fallback for all other task_ids.

score_task always returns both score_deterministic and score_judge as separate
fields. They are NEVER averaged. The caller decides which axis to report on.
The top-level `score` key equals score_deterministic when available, else
score_judge, for backward compatibility with sweep.py and certify.py.

The probe set in evaluation/probes/ is a separate measurement track used by
harness/runner.py to evaluate local-model quality during the degradation sweep.
Probes carry fixed answer keys for their own prompts and cannot score the 14
live agent tasks, which produce open-ended outputs unrelated to any probe's
expected values. Do not create a task->probe mapping. See docs/THREATS.md.
"""

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
    if "n*(n+1)*(2*n+1)/6" in text or "2*n+1" in text:
        hits += 1
    return round((hits / 3) * 10.0, 2)


class QualityEvaluator:
    """Scores CN-01 deterministically; all other tasks via cached LLM judge.

    score_task returns both score_deterministic and score_judge as separate
    fields so callers can report on either axis independently.
    """

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
        """Return score metadata.

        Always includes:
          score               -- primary score (0-10), deterministic if available
          method              -- how score was derived
          score_deterministic -- 0-10 from programmatic scorer, or null
          score_judge         -- 0-10 from LLM judge, or null
          judge_model         -- judge model used, or null
          judge_prompt_version
        """
        output_text = _extract_text(response_json)
        score_deterministic: float | None = None
        score_judge: float | None = None
        det_method: str | None = None
        judge_model: str | None = None
        judge_replayed: bool | None = None
        judge_cache_key: str | None = None

        if DETERMINISTIC_SCORERS.get(task_id) == "compute_numerical":
            score_deterministic = _score_cn01(output_text)
            det_method = "deterministic_programmatic"

        if score_deterministic is None:
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
            judge_result = self.judge_backend.model_call(
                messages=judge_messages,
                tools=None,
                temperature=0.0,
                seed=0,
                model=self.judge_model,
            )
            judge_text = _extract_text(judge_result.response_json)
            score_judge = self._parse_score(judge_text)
            judge_model = self.judge_model
            judge_replayed = judge_result.replayed
            judge_cache_key = judge_result.cache_key

        primary_score = score_deterministic if score_deterministic is not None else score_judge
        method = det_method if score_deterministic is not None else "llm_judge"

        return {
            "score": primary_score,
            "method": method,
            "score_deterministic": score_deterministic,
            "score_judge": score_judge,
            "judge_model": judge_model,
            "judge_prompt_version": self.judge_prompt_version if judge_model else None,
            "judge_replayed": judge_replayed,
            "judge_cache_key": judge_cache_key,
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
