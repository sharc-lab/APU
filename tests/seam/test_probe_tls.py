"""Unit tests for the TLS probe summarizer - no live network."""

from __future__ import annotations

from seam.tools.probe_tls import Attempt, ProbeResult, summarize


def _attempt(host: str, exit_code: int, *, mode: str = "plain", n: int = 1) -> Attempt:
    return Attempt(
        host=host,
        url=f"https://{host}/",
        mode=mode,
        attempt=n,
        exit_code=exit_code,
        http_code="200" if exit_code == 0 else "",
        elapsed_s=0.1,
        tls_handshake_failure=exit_code == 35,
    )


def _result(codes: list[int], host: str = "api.anthropic.com") -> ProbeResult:
    return ProbeResult(
        started_utc="t0",
        finished_utc="t1",
        attempts_per_target=len(codes),
        modes=["plain"],
        attempts=[_attempt(host, c, n=i + 1) for i, c in enumerate(codes)],
    )


class TestSummarize:
    def test_eight_of_eight_is_healthy(self) -> None:
        s = summarize(_result([0] * 8))
        assert s["verdict"]["label"] == "healthy"
        assert s["verdict"]["retry_required_before_spend"] is False

    def test_three_of_eight_is_severely_degraded(self) -> None:
        s = summarize(_result([0, 35, 35, 0, 35, 35, 0, 35]))
        assert s["verdict"]["label"] == "severely_degraded"
        assert s["verdict"]["retry_required_before_spend"] is True
        assert s["plain"]["api.anthropic.com"]["tls35"] == 5

    def test_five_of_eight_is_degraded(self) -> None:
        s = summarize(_result([0, 0, 0, 0, 0, 35, 35, 35]))
        assert s["verdict"]["label"] == "degraded"
        assert s["verdict"]["retry_required_before_spend"] is True

    def test_exit_codes_are_preserved_individually(self) -> None:
        codes = [0, 35, 28, 0, 35, 0, 7, 0]
        s = summarize(_result(codes))
        assert s["plain"]["api.anthropic.com"]["exit_codes"] == codes
        assert s["plain"]["api.anthropic.com"]["other_fail"] == 2  # 28 and 7
