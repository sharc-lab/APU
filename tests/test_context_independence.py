"""Assert that generated filler never contains probe expected-answer strings.

This is the foundational assumption of the degradation study: score changes
under context growth are attributable to the model's ability to use its own
knowledge under longer contexts, not to retrieval from filler. If filler
happened to contain an answer string, a score improvement at depth D would be
confounded with retrieval-augmented generation.

Tests check:
  - All six context depths used in the sweep.
  - Three seeds (0, 1, 42) to cover different filler starting offsets.
  - Both filler modes (labelled and unlabelled), since the filler text itself
    is the same in both; only the wrapper differs.
  - Expected values that are strings (exact and span_match probes). Schema and
    unit_test expected values are paths/dicts and are skipped.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
PROBES_FILE = REPO_ROOT / "evaluation" / "probes" / "prompts.jsonl"

DEPTHS = [0, 2_000, 8_000, 16_000, 32_000, 64_000]
SEEDS = [0, 1, 42]


def _load_probes() -> list[dict]:
    probes = []
    with open(PROBES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))
    return probes


def _string_expected_values(probe: dict) -> list[str]:
    """Return the set of string values we must not find in filler."""
    st = probe["scorer_type"]
    exp = probe["expected"]
    if st == "exact":
        return [str(exp)]
    if st == "span_match":
        terms: list[str] = []
        terms.extend(exp.get("required_all", []))
        for group in exp.get("required_any", []):
            terms.extend(group)
        return [t for t in terms if len(t) > 3]
    return []


@pytest.fixture(scope="module")
def probes():
    return _load_probes()


@pytest.fixture(scope="module")
def build_filler():
    from harness.context import build_filler as _build_filler
    return _build_filler


@pytest.mark.parametrize("depth", [d for d in DEPTHS if d > 0])
@pytest.mark.parametrize("seed", SEEDS)
def test_filler_does_not_contain_exact_answers(probes, build_filler, depth, seed):
    """No exact-scored probe's expected answer should appear in the filler text."""
    filler = build_filler(depth, seed=seed).lower()
    violations = []
    for probe in probes:
        for term in _string_expected_values(probe):
            if len(term) > 3 and term.lower() in filler:
                violations.append(
                    f"probe={probe['id']} term={term!r} found at depth={depth} seed={seed}"
                )
    assert not violations, (
        "Filler contains expected-answer strings — degradation results would be "
        "confounded with retrieval:\n" + "\n".join(violations)
    )


def test_zero_depth_returns_empty(build_filler):
    assert build_filler(0) == ""
    assert build_filler(0, seed=99) == ""


def test_filler_grows_with_depth(build_filler):
    sizes = [len(build_filler(d)) for d in [1_000, 4_000, 16_000]]
    assert sizes[0] < sizes[1] < sizes[2]


def test_filler_is_deterministic(build_filler):
    a = build_filler(8_000, seed=7)
    b = build_filler(8_000, seed=7)
    assert a == b


def test_different_seeds_differ(build_filler):
    a = build_filler(8_000, seed=0)
    b = build_filler(8_000, seed=1)
    assert a != b
