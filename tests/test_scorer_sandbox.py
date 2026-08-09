"""Verify that score_unit_test does not leak sensitive env vars to generated code.

This test must fail loudly if anyone widens the sandbox environment later.
The OPENAI_API_KEY canary is used because the repo's quick-start exports it
into the shell; it is the most realistic exfiltration target.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

PROBES_DIR = pathlib.Path(__file__).parent.parent / "evaluation" / "probes"


def _load_scorers():
    spec = importlib.util.spec_from_file_location(
        "probes_scorers", PROBES_DIR / "scorers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scorers():
    return _load_scorers()


TEST_COD_01 = str(PROBES_DIR / "tests" / "test_cod_01.py")
REF_COD_01 = (PROBES_DIR / "reference" / "ref_cod_01.py").read_text()

CANARY = "sk-CANARY-DO-NOT-LEAK"

LEAK_CODE = """\
import os
def reverse_words(s):
    raise SystemExit("LEAK:" + os.environ.get("OPENAI_API_KEY", "ABSENT"))
"""


def test_canary_not_leaked(scorers):
    """Generated code attempting to read OPENAI_API_KEY must not surface its value."""
    original = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = CANARY
    try:
        score, detail = scorers.score_unit_test(LEAK_CODE, TEST_COD_01)
        assert CANARY not in detail, (
            f"SANDBOX REGRESSION: OPENAI_API_KEY value appeared in scorer detail: {detail!r}"
        )
    finally:
        if original is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original


def test_canary_score_is_zero(scorers):
    """The leaking submission must score 0 (all tests fail/error)."""
    os.environ["OPENAI_API_KEY"] = CANARY
    try:
        score, detail = scorers.score_unit_test(LEAK_CODE, TEST_COD_01)
        assert score < 1.0, f"Leaking code unexpectedly scored {score}"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


def test_reference_passes(scorers):
    """Sanity: the reference implementation scores 1.0."""
    score, detail = scorers.score_unit_test(REF_COD_01, TEST_COD_01)
    assert score == 1.0, f"Reference scored {score} ({detail})"
    assert "4/4 passed" in detail


def test_safe_env_is_not_full_environ(scorers):
    """score_unit_test must build an explicit allowlist, not a copy of os.environ.

    This is a static check: we inject a sentinel into os.environ and confirm
    it does not appear in the subprocess output.
    """
    sentinel_key = "_HARNESS_SENTINEL_KEY_"
    sentinel_val = "SENTINEL_VALUE_MUST_NOT_LEAK"
    os.environ[sentinel_key] = sentinel_val
    try:
        score, detail = scorers.score_unit_test(
            f"""\
import os
def reverse_words(s):
    raise SystemExit("SENTINEL:" + os.environ.get({sentinel_key!r}, "ABSENT"))
""",
            TEST_COD_01,
        )
        assert sentinel_val not in detail, (
            f"SANDBOX REGRESSION: arbitrary env key leaked into detail: {detail!r}"
        )
    finally:
        os.environ.pop(sentinel_key, None)
