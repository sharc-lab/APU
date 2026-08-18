"""
Scorers for the Paper 1 quality axis.

Every scorer returns (score: float in [0,1], detail: str).
Deterministic scorers are pure functions of (output, expected) with NO model
calls, so they are stable across runs and reproduce on any machine. This is what
lets a score change be attributed to hardware/context degradation rather than
scorer variance.

The judge scorer is deliberately isolated here and returns score=None until
wired, so judge results are always reported in a separate column.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import pathlib

ROOT = pathlib.Path(__file__).parent


# ---------------------------------------------------------------- helpers

_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)

# Fullwidth digits U+FF10-FF19 and letters U+FF21-FF3A/FF41-FF5A -> ASCII
_FULLWIDTH = str.maketrans({
    **{chr(0xFF10 + i): str(i) for i in range(10)},
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},
    "\uff0c": ",", "\uff0e": ".", "\uff1a": ":", "\uff1b": ";",
    "\uff05": "%", "\uff08": "(", "\uff09": ")", "\uff0f": "/",
    "\u3001": ",", "\u3002": ".", "\u2212": "-", "\u2013": "-",
    "\u2014": "-", "\u201c": '"', "\u201d": '"',
})


def strip_fences(text: str) -> str:
    """Return the contents of the first fenced block, else the text unchanged."""
    m = _FENCE.search(text)
    return m.group(1) if m else text


def normalize_scalar(s: str) -> str:
    """
    Normalization for `exact`. Deliberately conservative: it removes formatting
    noise models add unprompted, but never rewrites the semantic content.
    """
    s = strip_fences(s).strip()
    # take the last non-empty line: models sometimes emit a stray preamble
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        s = lines[-1]
    s = s.strip()
    s = s.strip("`\"'")
    s = re.sub(r"^(the\s+)?(answer|result|final answer)\s*(is|:)\s*", "", s, flags=re.I)
    s = s.rstrip(".!")
    s = re.sub(r"\s+", "", s)          # collapse all whitespace
    # Qwen-family models frequently emit CJK fullwidth punctuation; map to ASCII
    s = s.translate(_FULLWIDTH)
    return s.lower()


def _num(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- exact

def score_exact(output: str, expected: str) -> tuple[float, str]:
    got = normalize_scalar(output)
    want = normalize_scalar(expected)
    if got == want:
        return 1.0, "exact match"

    # numeric equivalence: 134.0 == 134.00, 0.15% == 0.15 percent
    g_clean = got.replace("%", "").replace("percent", "")
    w_clean = want.replace("%", "").replace("percent", "")
    gn, wn = _num(g_clean), _num(w_clean)
    if gn is not None and wn is not None and abs(gn - wn) < 1e-9:
        # only accept if percent-ness agrees
        if ("%" in got or "percent" in got) == ("%" in want or "percent" in want):
            return 1.0, "numeric match"

    return 0.0, f"got={got!r} want={want!r}"


# ---------------------------------------------------------------- schema

def score_schema(output: str, expected_path: str) -> tuple[float, str]:
    import jsonschema

    schema = json.loads((ROOT / expected_path).read_text())
    raw = strip_fences(output).strip()

    # tolerate leading/trailing prose by extracting the outermost JSON object
    if not raw.startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        if i == -1 or j == -1 or j < i:
            return 0.0, "no JSON object found"
        raw = raw[i:j + 1]

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return 0.0, f"invalid JSON: {e.msg}"

    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        return 0.0, f"schema fail at {list(e.absolute_path)}: {e.message[:120]}"

    return 1.0, "valid"


# ---------------------------------------------------------------- span_match

def score_span_match(output: str, spec: dict) -> tuple[float, str]:
    """
    spec keys (all optional):
      required_all  : list[str]        every one must appear
      required_any  : list[list[str]]  each inner list: at least one must appear
      any_of_groups : list[list[str]]  each group: at least one member must appear
      forbidden     : list[str]        none may appear
    Score is the fraction of satisfied constraints; forbidden hits zero it out.
    """
    t = output.lower()
    checks, passed = 0, 0
    fails = []

    for term in spec.get("required_all", []):
        checks += 1
        if term.lower() in t:
            passed += 1
        else:
            fails.append(f"missing:{term}")

    for group in spec.get("required_any", []):
        checks += 1
        if any(x.lower() in t for x in group):
            passed += 1
        else:
            fails.append(f"missing_any:{group}")

    for group in spec.get("any_of_groups", []):
        checks += 1
        if any(x.lower() in t for x in group):
            passed += 1
        else:
            fails.append(f"missing_group:{group[:3]}...")

    for term in spec.get("forbidden", []):
        if term.lower() in t:
            return 0.0, f"forbidden present: {term}"

    if checks == 0:
        return 1.0, "no constraints"
    return passed / checks, ("ok" if not fails else ";".join(fails)[:200])


# ---------------------------------------------------------------- unit_test

def score_unit_test(output: str, test_rel_path: str, timeout: int = 30) -> tuple[float, str]:
    """
    Writes the model's code to candidate.py in a temp dir alongside the probe's
    pytest file, runs pytest, and scores the fraction of tests that pass.
    Fractional credit matters here: it distinguishes 'slightly wrong' from
    'collapsed', which is exactly the degradation signal we want.

    THREAT MODEL
    ------------
    This executes untrusted model-generated code. Two mitigations applied:

      1. SCRUBBED ENVIRONMENT. The child gets a minimal env, NOT the parent's.
         Without this the child inherits OPENAI_API_KEY / ANTHROPIC_API_KEY --
         the repo's own quick-start exports one -- giving generated code a
         direct exfiltration path.
      2. RESOURCE LIMITS. CPU seconds and address space are capped so a runaway
         or deliberately hostile generation cannot wedge the run. POSIX only;
         on Windows the timeout is the sole limit, which is why the sweep should
         run on the Linux box.

    Still NOT sandboxed: filesystem and network. Generated code can read files
    the user can read and open sockets. For a local model this is acceptable;
    before running cached completions from an untrusted source, containerize.
    """
    code = strip_fences(output)
    test_src = (ROOT / test_rel_path).read_text()

    # Minimal env: enough to find the interpreter, nothing sensitive.
    # On Windows, SystemRoot is required for Winsock DLL lookup (WinError 10106).
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/nonexistent",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "NO_PROXY": "*",
    }
    # SystemRoot/SystemDrive: Windows requires these to locate Winsock DLLs
    # (absent -> WinError 10106 in pytest subprocess).
    # TEMP/TMP: pytest writes its own temp files; without these it falls back to
    # cwd which may not be writable.
    # USERPROFILE is deliberately excluded — it leaks the local username path and
    # is not required for the subprocess to start or for pytest to run.
    for _win_key in ("SystemRoot", "SystemDrive", "TEMP", "TMP"):
        if _win_key in os.environ:
            safe_env[_win_key] = os.environ[_win_key]

    def _limits():  # pragma: no cover - POSIX only, runs in the child
        import resource
        for name, soft, hard in [
            ("RLIMIT_CPU", timeout, timeout),
            ("RLIMIT_AS", 2 << 30, 2 << 30),
            ("RLIMIT_CORE", 0, 0),
            ("RLIMIT_NPROC", 256, 256),
        ]:
            try:
                res = getattr(resource, name)
                resource.setrlimit(res, (soft, hard))
            except (AttributeError, ValueError, OSError, Exception):
                pass

    popen_extra = {}
    if os.name == "posix":
        popen_extra["preexec_fn"] = _limits

    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / "candidate.py").write_text(code)
        (d / "test_probe.py").write_text(test_src)
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-m", "pytest", "test_probe.py", "-q",
                 "--no-header", "-p", "no:cacheprovider"],
                cwd=d, capture_output=True, text=True, timeout=timeout,
                env=safe_env, **popen_extra,
            )
        except subprocess.TimeoutExpired:
            return 0.0, "timeout"

    tail = (r.stdout or "") + (r.stderr or "")
    m_p = re.search(r"(\d+) passed", tail)
    m_f = re.search(r"(\d+) failed", tail)
    m_e = re.search(r"(\d+) error", tail)
    np_ = int(m_p.group(1)) if m_p else 0
    nf = int(m_f.group(1)) if m_f else 0
    ne = int(m_e.group(1)) if m_e else 0
    total = np_ + nf + ne
    if total == 0:
        return 0.0, f"no tests ran: {tail.strip()[-160:]}"
    return np_ / total, f"{np_}/{total} passed"


# ---------------------------------------------------------------- judge

class JudgeNotConfigured(RuntimeError):
    pass


def score_judge(output: str, spec: dict, client=None) -> tuple[float | None, str]:
    """
    Judge scoring is OPTIONAL and reported separately. Returns (None, reason)
    when no client is configured so a run is never blocked on API access.
    """
    if client is None:
        return None, "judge not configured"
    raise JudgeNotConfigured("wire the judge client before enabling judge probes")


# ---------------------------------------------------------------- abstention

# Phrases that assert the *source material* (provided text/documents/context)
# is missing the needed information. Ordered longest-first so a more specific
# phrase matches before its prefix would.
_ABSTENTION_PHRASES: tuple[str, ...] = (
    "impossible to determine",
    "cannot be determined",
    "unable to determine",
    "cannot determine",
    "does not mention",
    "does not contain",
    "not contain",
    "not mentioned",
    "not provided",
    "not specified",
    "not stated",
    "not given",
    "no information",
    "no mention",
    "insufficient information",
    "insufficient data",
    "not present in",
    "not available in",
    "documents do not",
    "document does not",
    "text does not",
    "passage does not",
    "context does not",
    "provided text",   # "provided text does not..." — partial match is fine
)

# After an abstention phrase, if the model continues with answer-giving language
# followed by a specific value, the output is a hedge-then-fabrication.
# We look for these patterns in the text *after* the abstention phrase.
#
# Uses adversative conjunctions only (but/however/though/yet), not consequential
# ones (so/therefore/thus), because "so it is impossible to determine" is a
# continuation of the abstention, not a pivot to an answer.
#
# Also matches explicit answer-declaration patterns without a leading conjunction.
_HEDGE_THEN_ANSWER = re.compile(
    r"\b(?:but|however|though|yet|still)\b"
    r"[^.]{0,80}"
    r"\b(?:likely|probably|approximately|around|about|would be|it is|they are)\s*"
    r"(?!(?:impossible|unable|cannot|not\b))"   # exclude "it is impossible" etc.
    r"[\d\w]",
    re.I,
)
_DIRECT_ANSWER_AFTER = re.compile(
    r"\b(?:the answer is|answer would be|result is|value is|that is)\s*[\d\w]",
    re.I,
)


def classify_abstention(output: str) -> bool:
    """Return True if *output* is a genuine abstention from the source material.

    Genuine abstention: the model explicitly declares that the provided
    text/documents/context does not contain the information needed to answer,
    without then going on to provide a specific answer anyway.

    Not an abstention:
    - Confident wrong answers with no hedge language.
    - Outputs that contain an abstention phrase but then provide a specific
      value (hedge-then-fabrication): e.g. "I cannot determine from the text,
      but it is approximately 0.15".

    The line: an abstention phrase must be present AND the text after that
    phrase must not contain answer-giving language followed by a specific value.
    A purely explanatory refusal (no answer materialises after the hedge) counts
    as abstention even if lengthy.
    """
    t = output.lower()

    # Find the position of the earliest abstention phrase.
    first_pos = -1
    for phrase in _ABSTENTION_PHRASES:
        pos = t.find(phrase)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos

    if first_pos == -1:
        return False  # no abstention language at all

    # Check whether the model answered anyway after the abstention phrase.
    after = t[first_pos:]
    if _HEDGE_THEN_ANSWER.search(after) or _DIRECT_ANSWER_AFTER.search(after):
        return False  # hedge-then-answer: fabrication

    return True


def outcome_class(output: str | None, score: float | None, truncated: bool) -> str:
    """Classify a row into one of four mutually exclusive outcome classes.

    Classes:
      correct    score == 1.0
      abstained  model declared source material insufficient (no answer given)
      incorrect  model gave a specific wrong answer
      error      output is None (call failed)

    Note: abstention and correctness are orthogonal by design. An abstention on
    a probe whose artifact is *present* (truncated=False) is a failure to answer;
    an abstention where the artifact was *removed* (truncated=True) is correct
    behaviour. The score field carries the scorer's judgment; outcome_class
    carries the response type. Callers must condition analysis on `truncated` to
    interpret abstentions correctly.
    """
    if output is None:
        return "error"
    if score == 1.0:
        return "correct"
    if classify_abstention(output):
        return "abstained"
    return "incorrect"


# ---------------------------------------------------------------- dispatch

def score(probe: dict, output: str, judge_client=None):
    st = probe["scorer_type"]
    exp = probe["expected"]
    if st == "exact":
        return score_exact(output, exp)
    if st == "schema":
        return score_schema(output, exp)
    if st == "span_match":
        return score_span_match(output, exp)
    if st == "unit_test":
        return score_unit_test(output, exp)
    if st == "judge":
        return score_judge(output, exp, judge_client)
    raise ValueError(f"unknown scorer_type: {st}")
