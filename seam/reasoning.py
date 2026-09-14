"""Reasoning-mode discriminant for the AM-024 two-arm design.

Marker presence is **not** a valid discriminant. Qwen3 implements ``enable_thinking=false`` by
pre-filling an *empty* ``<think>\\n\\n</think>`` pair into the prompt, so a ``<think>`` substring
appears in the off-arm by construction - and appears again in the output whenever the runtime
echoes the prompt. Testing for the marker therefore reports "thinking is on" in both arms.

The discriminant is **non-empty thinking content**: the text between ``<think>`` and ``</think>``,
whitespace-stripped. The off-arm must produce none; the on-arm must produce some.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "THINKING_OCCURRED_MIN_NON_WS",
    "ReasoningSplit",
    "prompt_echo_status",
    "reasoning_verdict",
    "split_reasoning",
    "strip_prompt_prefix",
    "thinking_content",
    "thinking_occurred",
]

#: Non-whitespace chars inside a generated ``<think>`` block required to count as thinking.
#: The off-arm empty pair is ``\\n\\n`` (0 non-ws). Threshold 10 clears that construction
#: with margin for trivial whitespace/punctuation noise while staying far below real chains.
THINKING_OCCURRED_MIN_NON_WS: Final = 10

_THINK_BLOCK: Final = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK: Final = re.compile(r"<think>", re.IGNORECASE)
#: An unterminated block: the model started reasoning and hit the token cap before closing.
_UNCLOSED: Final = re.compile(r"<think>(?!.*?</think>)(.*)\Z", re.DOTALL | re.IGNORECASE)


class ReasoningSplit:
    """Reasoning content and visible answer, separated."""

    __slots__ = ("answer", "blocks", "truncated")

    def __init__(self, blocks: list[str], answer: str, truncated: bool) -> None:
        self.blocks = blocks
        self.answer = answer
        self.truncated = truncated

    @property
    def content(self) -> str:
        return "\n".join(b.strip() for b in self.blocks if b.strip()).strip()

    @property
    def has_content(self) -> bool:
        return bool(self.content)

    def to_dict(self) -> dict[str, object]:
        return {
            "thinking_chars": len(self.content),
            "thinking_has_content": self.has_content,
            "n_think_blocks": len(self.blocks),
            "thinking_truncated": self.truncated,
            "answer_chars": len(self.answer.strip()),
        }


def split_reasoning(text: str) -> ReasoningSplit:
    """Separate ``<think>`` blocks from the visible answer."""
    blocks = [m.group(1) for m in _THINK_BLOCK.finditer(text)]
    answer = _THINK_BLOCK.sub("", text)

    truncated = False
    unclosed = _UNCLOSED.search(answer)
    if unclosed:
        # A block opened but never closed is still reasoning content; counting it as answer text
        # would inflate the answer and hide that the cap was hit.
        truncated = True
        blocks.append(unclosed.group(1))
        answer = answer[: unclosed.start()]

    answer = _OPEN_THINK.sub("", answer)
    return ReasoningSplit(blocks, answer, truncated)


def thinking_content(text: str) -> str:
    """Convenience wrapper: the stripped reasoning content only."""
    return split_reasoning(text).content


def thinking_occurred(text: str, *, min_non_ws: int = THINKING_OCCURRED_MIN_NON_WS) -> bool:
    """True iff a think-block in ``text`` has more than ``min_non_ws`` non-whitespace chars.

    Marker presence alone is not evidence (empty pair is pre-filled into the off-arm prompt).
    """
    split = split_reasoning(text)
    for block in split.blocks:
        non_ws = sum(1 for ch in block if not ch.isspace())
        if non_ws > min_non_ws:
            return True
    return False


def strip_prompt_prefix(raw: str, prompt: str) -> tuple[str, bool]:
    """If ``raw`` begins with ``prompt``, return ``(raw[len(prompt):], True)``.

    D1: some pipelines return prompt+completion. Every text metric must strip first when so.
    """
    if prompt and raw.startswith(prompt):
        return raw[len(prompt) :], True
    return raw, False


def prompt_echo_status(raw: str, prompt: str, *, nonce: str) -> dict[str, object]:
    """D1 probe: does a high-entropy nonce embedded in the prompt reappear in the return?

    Returns whether the nonce appears, whether a prompt-prefix strip applied, and the
    post-strip text used for all further metrics.
    """
    appears = bool(nonce) and nonce in raw
    stripped, stripped_prefix = strip_prompt_prefix(raw, prompt)
    # If the whole prompt was not a prefix but the nonce still appears, strip up to/including
    # the first nonce occurrence only when it sits inside a leading prompt echo.
    return {
        "nonce": nonce,
        "nonce_appears_in_raw": appears,
        "pipeline": "prompt_plus_completion" if appears else "completion_only",
        "stripped_prompt_prefix": stripped_prefix,
        "text_for_metrics": stripped if appears else raw,
    }


def reasoning_verdict(*, off_text: str, on_text: str) -> dict[str, object]:
    """Judge whether ``enable_thinking`` produced a real contrast, in both directions.

    A flag that silently does nothing would make Arm 1 and Arm 2 the same experiment, so this
    gates **both** arms, not just the reasoning one. Uses :func:`thinking_occurred` (content
    >10 non-ws), not marker presence.
    """
    off_ok = not thinking_occurred(off_text)
    on_ok = thinking_occurred(on_text)
    off = split_reasoning(off_text)
    on = split_reasoning(on_text)

    if off_ok and on_ok:
        verdict, note = "pass", ""
    elif not off_ok and not on_ok:
        verdict, note = (
            "refused",
            "Neither direction behaved: the off-arm emitted reasoning content AND the on-arm "
            "emitted none. The flag is not controlling the model.",
        )
    elif not off_ok:
        verdict, note = (
            "refused",
            "The off-arm produced reasoning content despite the empty pre-filled block. Arm 1 is "
            "NOT a non-reasoning condition, so both arms would be reasoning arms.",
        )
    else:
        verdict, note = (
            "refused",
            "The on-arm produced no reasoning content. Arm 2 would not differ from Arm 1.",
        )

    return {
        "verdict": verdict,
        "note": note,
        "thinking_occurred_min_non_ws": THINKING_OCCURRED_MIN_NON_WS,
        "thinking_off_occurred": thinking_occurred(off_text),
        "thinking_on_occurred": thinking_occurred(on_text),
        "thinking_off": off.to_dict(),
        "thinking_on": on.to_dict(),
    }
