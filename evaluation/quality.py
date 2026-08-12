"""Per-task quality scoring for APU benchmark runs.

Scoring hierarchy (applied in order, first hit wins):
  1. Probe-backed deterministic: task_id is in the task_probe_map AND at least
     one of the mapped probes has scorer_type == span_match.  All span_match
     probes in the list are scored; the mean of their 0-10 scores becomes
     score_deterministic.  Per-probe scores are retained in probe_scores.
     Other scorer types (exact, schema, unit_test) have prompt-specific
     expected values and return null from _score_with_probe — they are listed
     for category coverage but do not contribute to score_deterministic.
  2. Deterministic programmatic: hard-coded scorer for CN-01.
  3. LLM judge: fallback when score_deterministic is None.

score_task always returns both score_deterministic and score_judge as separate
fields. They are NEVER averaged. The top-level `score` key equals
score_deterministic when available, else score_judge, for backward
compatibility with sweep.py and certify.py.

probe_scores: dict[probe_id -> {score, detail}] is included in the return dict
whenever the task is in the task_probe_map.  Entries with score=null are probes
whose scorer_type is not span_match; they are listed for coverage bookkeeping.
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


def _load_task_probe_map() -> dict[str, list[dict]]:
    """Load task_map.yaml; return {} when absent.

    Returns dict mapping task_id -> list of probe dicts (full probe objects
    from prompts.jsonl).  Only probe IDs that exist in prompts.jsonl are kept.
    """
    if not TASK_MAP_PATH.exists():
        return {}
    try:
        import yaml
        raw = yaml.safe_load(TASK_MAP_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    probes: dict[str, dict] = {}
    jsonl = PROBES_DIR / "prompts.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                p = json.loads(line)
                probes[p["id"]] = p

    mapping: dict[str, list[dict]] = {}
    for task_id, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        ids = entry.get("probe_ids", [])
        if isinstance(ids, str):
            ids = [ids]
        resolved = [probes[pid] for pid in ids if pid in probes]
        if resolved:
            mapping[task_id] = resolved
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

    For tasks in task_probe_map, all span_match probes in the mapped list are
    scored and their mean becomes score_deterministic.  Per-probe scores appear
    in probe_scores.  Non-span_match probes in the list are included in
    probe_scores with score=null (for category coverage bookkeeping) but do not
    contribute to score_deterministic.  The judge runs only when
    score_deterministic is None.
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
        self._task_probe_map: dict[str, list[dict]] = _load_task_probe_map()

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
        expected values are prompt-specific and cannot evaluate outputs from a
        different task prompt.  Returns (None, reason) for non-span_match types
        so callers can record them in probe_scores without counting as a score.
        """
        scorer_type = probe.get("scorer_type")
        if scorer_type != "span_match":
            return None, f"scorer_type={scorer_type!r} not applicable to task outputs"
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
          score_deterministic -- 0-10 mean of span_match probe scores, or null
          score_judge         -- 0-10 from LLM judge, or null
          probe_scores        -- dict[probe_id -> {score, detail}] when mapped, else null
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
        probe_scores: dict[str, Any] | None = None

        # --- Deterministic path 1: programmatic hard-coded scorer (CN-01) ---
        if DETERMINISTIC_SCORERS.get(task_id) == "compute_numerical":
            score_deterministic = _score_cn01(output_text)
            det_method = "deterministic_programmatic"

        # --- Deterministic path 2: probe-backed span_match ---
        elif task_id in self._task_probe_map:
            probe_list = self._task_probe_map[task_id]
            probe_scores = {}
            span_scores: list[float] = []
            for probe in probe_list:
                s, detail = self._score_with_probe(output_text, probe)
                probe_scores[probe["id"]] = {"score": s, "detail": detail}
                if s is not None:
                    span_scores.append(s)
            if span_scores:
                score_deterministic = round(sum(span_scores) / len(span_scores), 2)
                det_method = (
                    f"probe_span_match:mean({len(span_scores)}/{len(probe_list)})"
                )

        # --- Judge path (runs only when no deterministic score is available) ---
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
            "probe_scores": probe_scores,
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
