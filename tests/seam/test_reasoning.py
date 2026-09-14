"""AM-024 reasoning discriminant: content, not marker presence."""

from __future__ import annotations

from seam.reasoning import (
    THINKING_OCCURRED_MIN_NON_WS,
    prompt_echo_status,
    reasoning_verdict,
    split_reasoning,
    strip_prompt_prefix,
    thinking_content,
    thinking_occurred,
)

# What the Qwen3 template actually produces in the off-arm: an EMPTY pre-filled pair.
OFF_TEXT = "<think>\n\n</think>\n\nThe scheduler compares predicted latency against the deadline."
ON_TEXT = (
    "<think>\nI should weigh prefill against decode, then compare to D.\n</think>\n\n"
    "The scheduler compares predicted latency against the deadline."
)


def test_empty_prefilled_block_is_not_thinking_content() -> None:
    """The off-arm marker must not be mistaken for reasoning."""
    assert thinking_content(OFF_TEXT) == ""
    assert split_reasoning(OFF_TEXT).has_content is False


def test_real_reasoning_is_detected() -> None:
    split = split_reasoning(ON_TEXT)
    assert split.has_content is True
    assert "prefill" in split.content
    assert split.answer.strip().startswith("The scheduler")


def test_verdict_passes_only_when_both_directions_behave() -> None:
    assert reasoning_verdict(off_text=OFF_TEXT, on_text=ON_TEXT)["verdict"] == "pass"


def test_off_arm_that_reasons_anyway_is_refused() -> None:
    """This is the failure that would silently make Arm 1 a reasoning arm too."""
    result = reasoning_verdict(off_text=ON_TEXT, on_text=ON_TEXT)
    assert result["verdict"] == "refused"
    assert "NOT a non-reasoning condition" in str(result["note"])


def test_on_arm_that_does_not_reason_is_refused() -> None:
    result = reasoning_verdict(off_text=OFF_TEXT, on_text=OFF_TEXT)
    assert result["verdict"] == "refused"
    assert "no reasoning content" in str(result["note"])


def test_both_directions_broken_is_refused() -> None:
    result = reasoning_verdict(off_text=ON_TEXT, on_text=OFF_TEXT)
    assert result["verdict"] == "refused"
    assert "Neither direction" in str(result["note"])


def test_unclosed_block_counts_as_truncated_reasoning() -> None:
    """Hitting the token cap mid-thought is reasoning, not answer text."""
    split = split_reasoning("<think>\nStill working through the tradeoff and I ran out of room")
    assert split.truncated is True
    assert split.has_content is True
    assert split.answer.strip() == ""


def test_multiple_blocks_are_concatenated() -> None:
    split = split_reasoning("<think>first</think>mid<think>second</think>tail")
    assert len(split.blocks) == 2
    assert "first" in split.content and "second" in split.content
    assert split.answer.replace(" ", "") == "midtail"


def test_empty_pair_is_below_thinking_occurred_threshold() -> None:
    assert THINKING_OCCURRED_MIN_NON_WS == 10
    assert thinking_occurred(OFF_TEXT) is False
    assert thinking_occurred(ON_TEXT) is True
    # Short non-empty noise below threshold must not count.
    assert thinking_occurred("<think>\nabc\n</think>\n\nanswer") is False


def test_prompt_echo_strip() -> None:
    prompt = "PREFIX_NONCE_abc123XYZ999_end of prompt"
    raw = prompt + "The answer is 42."
    status = prompt_echo_status(raw, prompt, nonce="abc123XYZ999")
    assert status["pipeline"] == "prompt_plus_completion"
    assert status["stripped_prompt_prefix"] is True
    assert status["text_for_metrics"] == "The answer is 42."
    stripped, ok = strip_prompt_prefix(raw, prompt)
    assert ok and stripped == "The answer is 42."


def test_completion_only_pipeline() -> None:
    prompt = "PREFIX_NONCE_abc123XYZ999_end"
    raw = "The answer is 42."
    status = prompt_echo_status(raw, prompt, nonce="abc123XYZ999")
    assert status["pipeline"] == "completion_only"
    assert status["nonce_appears_in_raw"] is False
