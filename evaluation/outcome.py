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

# ---------------------------------------------------------------- constants

CORRECT = "CORRECT"
REFUSED = "REFUSED"
FABRICATED = "FABRICATED"
UNCLASSIFIABLE = "UNCLASSIFIABLE"

# Arm-2 of selfreport_arms instructs the model to respond with this exact
# string when context is insufficient. Treat it as a refusal.
_INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

# Matches the last standalone numeric token (integer or decimal, optional
# leading minus).  Word-boundary anchors prevent "120" matching inside "1200".
_LAST_NUMERIC = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)")


_OUTCOME_FIELDS = ("outcome_class", "classification_method", "format_compliant")


def normalize_result_row(d: dict) -> dict:
    """Return a copy of d with all outcome fields present (None if absent).

    Backward compat: rows written before the four-way classifier was wired in
    do not carry outcome_class, classification_method, or format_compliant.
    Analysis code should call this when loading rows from JSONL files so that
    old rows and new rows present the same dict shape. Old rows get None —
    never a guessed value.
    """
    out = dict(d)
    for field in _OUTCOME_FIELDS:
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
    if classify_abstention(output):
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
    if scorer_type == "exact" and not _has_abstention_language(output):
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

    # ── FABRICATED: everything else ───────────────────────────────────────
    return {
        "outcome_class": FABRICATED,
        "format_compliant": False if scorer_type == "exact" else None,
        "classification_method": "score",
    }
