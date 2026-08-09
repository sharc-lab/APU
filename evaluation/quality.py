"""Per-task quality scoring for APU benchmark runs.

Scoring hierarchy (applied in order, first hit wins):
  1. Probe-backed deterministic: task_id is in the task_probe_map AND the
     mapped probe has a scorer_type that can meaningfully evaluate open-ended
     task outputs (span_match only for now — exact/schema/unit_test scorers are
     tied to specific probe prompts and cannot be applied to arbitrary outputs).
  2. Deterministic programmatic: hard-coded scorer for CN-01.
  3. LLM judge: fallback for all other task_ids.

score_task always returns both score_deterministic and score_judge as separate
fields. They are NEVER averaged. The caller decides which axis to report on.
The top-level `score` key equals score_deterministic when available, else
score_judge, for backward compatibility with sweep.py and certify.py.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from harness.backends.cloud_openai import CloudOpenAIBackend
from harness.replay import ReplayMode

PROBES_DIR = Path(__file__).parent.parent / "evaluation" / "probes"
TASK_MAP_PATH = Path(__file__).parent / "probes" / "task_map.yaml"

DETERMINISTIC_SCORERS = {
    "CN-01": "compute_numerical",
}


def _load_probe_scorers():
    """Load evaluation/probes/scorers.py via importlib (avoids sys.path mutation)."""
    spec = importlib.util.spec_from_file_location(
        "probes_scorers", PROBES_DIR / "scorers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_task_probe_map() -> dict[str, dict]:
    """Load task_map.yaml if it exists; return {} otherwise."""
    if not TASK_MAP_PATH.exists():
        return {}
    try:
        import yaml
        raw = yaml.safe_load(TASK_MAP_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    # Load the probe index keyed by probe_id
    probes: dict[str, dict] = {}
    jsonl = PROBES_DIR / "prompts.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                p = json.loads(line)
                probes[p["id"]] = p

    mapping: dict[str, dict] = {}
    for task_id, entry in raw.items():
        probe_id = entry.get("probe_id") if isinstance(entry, dict) else None
        if probe_id and probe_id in probes:
            mapping[task_id] = probes[probe_id]
    return mapping


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


def _probe_score_to_10(score: float | None) -> float | None:
    """Convert 0-1 probe score to 0-10 scale to match judge scale."""
    if score is None:
        return None
    return round(score * 10.0, 2)


class QualityEvaluator:
    """Scores deterministic tasks directly and open-ended tasks via cached LLM judge.

    When task_probe_map is populated (loaded from evaluation/probes/task_map.yaml),
    tasks with probe mappings are scored deterministically using
    evaluation/probes/scorers.py for span_match probes. score_task returns both
    score_deterministic and score_judge as separate fields.
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
        self._probe_scorers = None
        self._task_probe_map: dict[str, dict] = _load_task_probe_map()

    def _get_probe_scorers(self):
        if self._probe_scorers is None:
            self._probe_scorers = _load_probe_scorers()
        return self._probe_scorers

    @staticmethod
    def _load_prompt(version: str) -> str:
        prompt_path = Path(__file__).parent / "judge_prompts" / f"{version}.md"
        return prompt_path.read_text(encoding="utf-8")

    def _score_with_probe(self, output_text: str, probe: dict) -> tuple[float | None, str]:
        """Apply a probe scorer to an open-ended task output.

        Only span_match is supported for task outputs: exact/schema/unit_test
        scorers are tied to specific probe prompts and cannot meaningfully
        evaluate outputs from a different task prompt.
        """
        scorer_type = probe.get("scorer_type")
        if scorer_type != "span_match":
            return None, f"probe scorer_type={scorer_type!r} not applicable to task outputs"
        scorers = self._get_probe_scorers()
        score_01, detail = scorers.score_span_match(output_text, probe["expected"])
        return _probe_score_to_10(score_01), detail

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
          score_deterministic -- 0-10 from probe/programmatic scorer, or null
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

        # --- Deterministic path 1: programmatic hard-coded scorer (CN-01) ---
        if DETERMINISTIC_SCORERS.get(task_id) == "compute_numerical":
            score_deterministic = _score_cn01(output_text)
            det_method = "deterministic_programmatic"

        # --- Deterministic path 2: probe-backed span_match ---
        elif task_id in self._task_probe_map:
            probe = self._task_probe_map[task_id]
            s, detail = self._score_with_probe(output_text, probe)
            if s is not None:
                score_deterministic = s
                det_method = f"probe_span_match:{probe['id']}"

        # --- Judge path (runs when no deterministic score, or always for dual-reporting) ---
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
