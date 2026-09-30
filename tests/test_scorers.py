"""Unit tests for evaluation/probes/scorers.py, confirming every R1b/Q0 scorer is deterministic string/JSON matching
with no model call, and covering the 2026-09-29 self_report scoring-artifact fix.

A1 controls (per the standing instruction): feed the scorer (i) the expected answer verbatim, (ii) the expected
answer wrapped in the self_report format, (iii) a plausible wrong answer. All three must score as expected, for
each deterministic scorer type used by this evaluation (exact, span_match, schema)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "probes"))
import scorers as sc  # noqa: E402


# ---------------------------------------------------------------------------------------------------- determinism audit
def test_score_dispatch_has_no_model_call_for_deterministic_types():
    """score()'s exact/schema/span_match/unit_test branches take no judge_client and cannot reach score_judge --
    confirmed by reading the dispatch table itself, not assumed. judge_client is only ever passed to score_judge,
    the one scorer type this module's own docstring says returns score=None until wired."""
    import inspect
    src = inspect.getsource(sc.score)
    # every branch except "judge" must not reference judge_client
    for line in src.splitlines():
        if "judge_client" in line and "score_judge" not in line and "def score" not in line:
            assert False, f"a non-judge branch references judge_client: {line}"


def test_score_judge_returns_none_until_wired():
    score, detail = sc.score_judge("anything", "anything", client=None)
    assert score is None


# ---------------------------------------------------------------------------------------------------- exact: the scorer_type every art_* probe uses
def test_exact_scores_verbatim_expected_as_correct():
    probe = {"scorer_type": "exact", "expected": "51847"}
    score, detail = sc.score(probe, "51847")
    assert score == 1.0


def test_exact_scores_self_report_wrapped_expected_as_correct():
    """The 2026-09-29 fix: R1b's self_report arm asks the model to preface its answer with 'AVAILABLE: yes|no',
    which previously made an exactly-correct answer score 0.0 (real case: baseline '51847' scored 1.0, self_report
    'AVAILABLE: yes 51847' scored 0.0, same model, same probe, same ratio)."""
    probe = {"scorer_type": "exact", "expected": "51847"}
    for wrapped in ("AVAILABLE: yes 51847", "AVAILABLE: yes\n51847", "AVAILABLE:yes12.4".replace("12.4", "51847"),
                    "AVAILABLE: no 51847", "available: YES 51847"):
        score, detail = sc.score(probe, wrapped)
        assert score == 1.0, f"{wrapped!r} did not score 1.0: {detail}"


def test_exact_scores_plausible_wrong_answer_as_incorrect():
    probe = {"scorer_type": "exact", "expected": "51847"}
    score, detail = sc.score(probe, "51846")
    assert score == 0.0


def test_exact_wrong_answer_still_wrong_after_self_report_wrapping():
    probe = {"scorer_type": "exact", "expected": "51847"}
    score, detail = sc.score(probe, "AVAILABLE: yes 51846")
    assert score == 0.0


def test_strip_available_prefix_is_a_noop_on_baseline_style_output():
    assert sc.strip_available_prefix("51847") == "51847"
    assert sc.strip_available_prefix("The answer is 51847") == "The answer is 51847"


def test_strip_available_prefix_handles_the_real_observed_formats():
    """Exact strings pulled from real R1b rows on 2026-09-29 (evo-t2s, qwen3-8b, art_01/art_02 at ratio 1.20)."""
    assert sc.strip_available_prefix("AVAILABLE: yes 51847") == "51847"
    assert sc.strip_available_prefix("AVAILABLE:yes12.4") == "12.4"


# ---------------------------------------------------------------------------------------------------- span_match (rag_* probes)
def test_span_match_verbatim_and_wrapped_and_wrong():
    probe = {"scorer_type": "span_match", "expected": {"required_all": ["meridian-3"], "required_any": [],
             "forbidden": ["meridian-4 still"], "any_of_groups": [["original firmware", "retains"]]}}
    verbatim = "Meridian-3, because it retains the original firmware."
    wrapped = "AVAILABLE: yes\nMeridian-3, because it retains the original firmware."
    wrong = "Meridian-4 still shows drift."
    score_v, _ = sc.score(probe, verbatim)
    score_w, _ = sc.score(probe, wrapped)
    score_x, _ = sc.score(probe, wrong)
    assert score_v == 1.0
    assert score_w == 1.0
    assert score_x < 1.0


# ---------------------------------------------------------------------------------------------------- schema
def test_score_schema_verbatim_and_wrapped_and_wrong(tmp_path, monkeypatch):
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(__import__("json").dumps(schema), encoding="utf-8")
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    probe = {"scorer_type": "schema", "expected": "schema.json"}
    verbatim = '{"answer": "51847"}'
    wrapped = 'AVAILABLE: yes\n{"answer": "51847"}'
    wrong = '{"wrong_key": "51847"}'
    score_v, _ = sc.score(probe, verbatim)
    score_w, _ = sc.score(probe, wrapped)
    score_x, _ = sc.score(probe, wrong)
    assert score_v == 1.0
    assert score_w == 1.0
    assert score_x == 0.0
