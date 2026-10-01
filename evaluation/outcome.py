"""
Four-way outcome classifier for probe results.

This module extends the existing three-class scheme in
evaluation/probes/scorers.py (correct/abstained/incorrect/error) with a
distinct UNCLASSIFIABLE class for rows where the model never produced
output, and adds format_compliant tracking so format failures are
separated from arithmetic errors.

Cannot modify evaluation/probes/scorers.py (probe track is frozen).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

# Load scorers via file path to avoid triggering evaluation/__init__.py,
# which imports evaluation.quality (requires openai package not always present).
_spec = importlib.util.spec_from_file_location(
    "scorers",
    Path(__file__).parent / "probes" / "scorers.py",
)
_scorers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorers)

classify_abstention = _scorers.classify_abstention
normalize_scalar = _scorers.normalize_scalar
_ABSTENTION_PHRASES = _scorers._ABSTENTION_PHRASES
_HEDGE_THEN_ANSWER = _scorers._HEDGE_THEN_ANSWER
_DIRECT_ANSWER_AFTER = _scorers._DIRECT_ANSWER_AFTER
_num = _scorers._num

# ---------------------------------------------------------------- constants

CORRECT = "CORRECT"
REFUSED = "REFUSED"
FABRICATED = "FABRICATED"
UNCLASSIFIABLE = "UNCLASSIFIABLE"
REJECTED = "REJECTED"

# Arm-2 of selfreport_arms instructs the model to respond with this exact
# string when context is insufficient. Treat it as a refusal.
_INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

# Matches the last standalone numeric token (integer or decimal, optional
# leading minus).  Word-boundary anchors prevent "120" matching inside "1200".
_LAST_NUMERIC = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)")

# ── supplementary refusal detection (2026-10-01 kappa-study fix) ───────────
# evaluation/probes/scorers.py's _ABSTENTION_PHRASES / classify_abstention
# cannot be modified (probe track is frozen -- see module docstring), but the
# 150-row kappa-agreement study (results/labeling/KAPPA_STUDY_NOTE.md) found
# two real, recurring refusal phrasings that list misses and that this layer
# is free to add on top of it:
#
# 1. The self_report arm's own designed unavailability signal ("AVAILABLE:
#    no", with or without a trailing comma/explanation) is a refusal by that
#    arm's protocol regardless of generic abstention phrase matching.
# 2. A handful of ordinary English refusal phrasings ("not present" bare,
#    "does not include", "does not specify", "not possible to determine")
#    that are near-synonyms of phrases already in scorers.py's list but not
#    themselves present in it.
_AVAILABLE_NO_RE = re.compile(r"^\s*AVAILABLE\s*:\s*no\b", re.I)
_AVAILABLE_NO_STRIP_RE = re.compile(r"^\s*AVAILABLE\s*:\s*no\s*[,:]?\s*", re.I)

_EXTRA_ABSTENTION_PHRASES: tuple[str, ...] = (
    "not present",          # scorers.py only has the narrower "not present in"
    "does not include",
    "does not specify",
    "not possible to determine",
)


_OUTCOME_FIELDS = ("outcome_class", "classification_method", "format_compliant")

# Run-identity and stop-reason fields added 2026-09-14.  Old rows lack them;
# normalize_result_row fills them with None so analysis code sees a uniform shape.
_IDENTITY_FIELDS = ("git_sha", "run_seed", "hostname", "operator", "done_reason")

# TTFT fields added 2026-09-14.  ttft_ms is None for replayed and non-streaming
# rows; ttft_source records why.  Any TTFT analysis MUST filter on
# ttft_source == "streamed" — other source values carry no valid measurement.
_TTFT_FIELDS = ("ttft_ms", "ttft_source")

# Thinking-suppression verification field added 2026-09-14.  thinking_chars is
# the total length of message.thinking content accumulated across all streamed
# chunks.  0 = suppression succeeded; >0 = model produced a thinking phase
# despite the flag; None = cache hit or row predates this field.
_THINKING_FIELDS = ("thinking_chars",)

# llama-server session metadata added 2026-09-16.  Present on rows produced via
# LlamaServerSession; None on Ollama rows and all rows written before this date.
# server_session_id links rows to the session whose R1 overflow probe verified
# --context-shift.  n_ctx_slot is ground truth for effective context size.
_SESSION_FIELDS = (
    "server_session_id",
    "n_ctx_slot",
    "build_id",
    "backend",
    "platform",
    "context_shift_probe_result",
)

# Replay and context-overflow fields added 2026-09-16.
_REPLAY_FIELDS = ("replayed",)
_CONTEXT_OVERFLOW_FIELDS = ("context_size_exceeded", "n_prompt_tokens", "n_ctx")

# Filler calibration method added 2026-09-21. Values:
#   "ollama_prompt_eval"    — Ollama count_fn (prompt_eval_count, Blade14 arm)
#   "llamaserver_tokenize"  — llama-server /tokenize endpoint (evo-t2s arm)
#   "heuristic"             — char-only estimate (depth=0 or uncalibrated)
_COUNT_METHOD_FIELDS = ("count_method",)


def normalize_result_row(d: dict) -> dict:
    """Return a copy of d with all schema-tracked fields present (None if absent).

    Backward compat: rows written before the four-way classifier, before the
    run-identity fields, before the TTFT/thinking fields, before the
    llama-server session metadata fields, or before count_method was added will
    be missing some of these keys.  Analysis code MUST call this when loading
    rows from JSONL files so that old and new rows present the same dict shape.
    Missing fields are filled with None — never a guessed value.
    """
    out = dict(d)
    for field in (
        _OUTCOME_FIELDS
        + _IDENTITY_FIELDS
        + _TTFT_FIELDS
        + _THINKING_FIELDS
        + _SESSION_FIELDS
        + _REPLAY_FIELDS
        + _CONTEXT_OVERFLOW_FIELDS
        + _COUNT_METHOD_FIELDS
    ):
        out.setdefault(field, None)
    return out


def _has_abstention_language(text: str) -> bool:
    """True if any abstention phrase appears in text (ignoring hedge correction)."""
    t = text.lower()
    return any(phrase in t for phrase in _ABSTENTION_PHRASES)


def _extract_last_numeric_token(text: str) -> str | None:
    """Return the last standalone numeric token in text, or None."""
    matches = _LAST_NUMERIC.findall(text)
    return matches[-1] if matches else None


def _is_available_no_sentinel(output: str) -> bool:
    """True if output opens with the self_report arm's own "AVAILABLE: no"
    unavailability signal AND whatever follows it is either nothing (bare
    sentinel, e.g. "AVAILABLE: no" / "AVAILABLE: no,") or more abstention
    language, not a concrete value. Observed in the kappa-agreement sample
    (results/labeling/KAPPA_STUDY_NOTE.md): "AVAILABLE: no, 0" and
    "AVAILABLE: no \n0" are genuine fabrications (the model declares
    unavailability and then guesses a number anyway), so the bare-"no"
    protocol signal must NOT be treated as refusal when a concrete answer
    follows it -- only when nothing, or only more refusal language, does."""
    if not _AVAILABLE_NO_RE.match(output):
        return False
    tail = _AVAILABLE_NO_STRIP_RE.sub("", output, count=1).strip().strip(".")
    if not tail:
        return True  # bare sentinel, nothing follows
    t = tail.lower()
    if not any(phrase in t for phrase in _ABSTENTION_PHRASES + _EXTRA_ABSTENTION_PHRASES):
        return False  # concrete content follows "AVAILABLE: no" -- fabrication, not refusal
    return not (_HEDGE_THEN_ANSWER.search(t) or _DIRECT_ANSWER_AFTER.search(t))


def _has_extra_abstention_language(output: str) -> bool:
    """classify_abstention() extended with _EXTRA_ABSTENTION_PHRASES (see the
    comment above that list for why these live here instead of in the frozen
    scorers.py). Same earliest-phrase-then-hedge-check logic as
    classify_abstention(), just over the combined phrase set."""
    t = output.lower()
    first_pos = -1
    for phrase in _ABSTENTION_PHRASES + _EXTRA_ABSTENTION_PHRASES:
        pos = t.find(phrase)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos
    if first_pos == -1:
        return False
    after = t[first_pos:]
    if _HEDGE_THEN_ANSWER.search(after) or _DIRECT_ANSWER_AFTER.search(after):
        return False
    return True


def _embedded_exact_match(output: str, expected: str) -> bool:
    """Recovers a format-noncompliant CORRECT for non-numeric expected values
    (part numbers, version strings, proper nouns, etc.) that the numeric-only
    _extract_last_numeric_token path can never catch -- e.g. output
    "AVAILABLE: yes, PN-38901" against expected "PN-38901" (see
    KAPPA_STUDY_NOTE.md mechanism 2). Guarded to non-numeric expected values
    only: a purely-numeric expected value is already handled by the last-
    numeric-token path, and a plain substring check on digits is unsafe
    (expected "120" is a substring of output "1200", a real wrong answer)."""
    if _num(expected) is not None:
        return False
    norm_expected = re.sub(r"\s+", "", expected.strip().lower())
    if len(norm_expected) < 3:
        return False
    norm_output = re.sub(r"\s+", "", output.lower())
    return norm_expected in norm_output


def classify(
    *,
    output: str | None,
    expected: str,
    scorer_type: str,
    score: float | None,
    done_reason: str | None = None,
) -> dict:
    """Classify one probe row into the four-way outcome scheme.

    Returns a dict with keys:
      outcome_class  : one of CORRECT / REFUSED / FABRICATED / UNCLASSIFIABLE
      format_compliant : bool | None
          True  — score==1.0 (scorer accepted the output as-is)
          False — last numeric token matched but surrounding text violated the
                  "ONLY the final answer" instruction
          None  — not applicable (non-exact scorer, REFUSED, UNCLASSIFIABLE)
      classification_method : str  ("score", "last_token", "refused_sentinel",
                                    "refused_abstention", "unclassifiable")
    """
    # ── UNCLASSIFIABLE: no output produced ───────────────────────────────
    if output is None or output == "":
        return {
            "outcome_class": UNCLASSIFIABLE,
            "format_compliant": None,
            "classification_method": "unclassifiable",
        }
    if done_reason in ("length", "budget_exhausted"):
        # output may be non-empty but was cut mid-generation; treat as
        # unclassifiable regardless.
        return {
            "outcome_class": UNCLASSIFIABLE,
            "format_compliant": None,
            "classification_method": "unclassifiable",
        }

    # ── CORRECT: scorer accepted ──────────────────────────────────────────
    # Score is checked before REFUSED so that a scorer-correct output that
    # happens to contain abstention language (rare, but possible) is not
    # mis-labelled as REFUSED.
    if score == 1.0:
        return {
            "outcome_class": CORRECT,
            "format_compliant": True,
            "classification_method": "score",
        }

    # ── REFUSED: sentinel or genuine abstention ───────────────────────────
    if _INSUFFICIENT_CONTEXT in output:
        return {
            "outcome_class": REFUSED,
            "format_compliant": None,
            "classification_method": "refused_sentinel",
        }
    if _is_available_no_sentinel(output):
        return {
            "outcome_class": REFUSED,
            "format_compliant": None,
            "classification_method": "refused_available_no",
        }
    if classify_abstention(output) or _has_extra_abstention_language(output):
        return {
            "outcome_class": REFUSED,
            "format_compliant": None,
            "classification_method": "refused_abstention",
        }

    # ── CORRECT (format-non-compliant): last numeric token matches expected
    # Only applicable to exact scorers where expected is a number.
    # Skip when abstention language is present in the output: a hedge-then-answer
    # ("cannot determine, but approximately 0.15") that happens to end in the
    # expected number is FABRICATED, not a format failure.
    if scorer_type == "exact" and not (_has_abstention_language(output) or _has_extra_abstention_language(output)):
        last_tok = _extract_last_numeric_token(output)
        if last_tok is not None:
            norm_expected = normalize_scalar(expected)
            norm_token = normalize_scalar(last_tok)
            if norm_token == norm_expected:
                # Correct arithmetic, wrong format (showed working etc.)
                return {
                    "outcome_class": CORRECT,
                    "format_compliant": False,
                    "classification_method": "last_token",
                }
        if _embedded_exact_match(output, expected):
            # Correct non-numeric answer (part number, version string, proper
            # noun, ...) embedded in a format-noncompliant output, e.g.
            # "AVAILABLE: yes, PN-38901" against expected "PN-38901".
            return {
                "outcome_class": CORRECT,
                "format_compliant": False,
                "classification_method": "embedded_match",
            }

    # ── FABRICATED: everything else ───────────────────────────────────────
    return {
        "outcome_class": FABRICATED,
        "format_compliant": False if scorer_type == "exact" else None,
        "classification_method": "score",
    }
