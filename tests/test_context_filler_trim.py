"""harness/context.py filler trim (2026-11-07: the 8192 live start printed 188/150 tok lines alternating). The trim must
stop on a repeated length, build each filler once, and return exactly what the old trim returned."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import context as C  # noqa: E402


def _old_trim(raw, target_tokens, count_fn, max_iter=8):
    """The trim as it was before 2026-11-07 (prints removed), for the byte-identity check."""
    target_chars = min(int(target_tokens * C._CHARS_PER_TOKEN), len(raw))
    filler = raw[:target_chars].strip()
    if count_fn is None:
        return filler
    best, best_err = filler, float("inf")
    for _ in range(max_iter):
        actual = count_fn(filler)
        if actual <= 0:
            break
        err = abs(actual - target_tokens) / max(target_tokens, 1)
        if err < best_err:
            best, best_err = filler, err
        if err <= 0.02:
            return filler
        new_chars = min(int(len(filler) * target_tokens / actual), len(raw))
        if new_chars == len(filler):
            break
        filler = raw[:new_chars].strip()
    return best


def _words(s):
    return round(len(s.split()) * 1.37)


@pytest.fixture(autouse=True)
def _clear_cache():
    C._FILLER_CACHE.clear()
    yield
    C._FILLER_CACHE.clear()


def test_two_length_oscillation_stops_at_the_repeat(capsys):
    raw = "x" * 20_000
    target = 150
    first = int(target * C._CHARS_PER_TOKEN)
    calls = []

    def flip(s):     # long text counts double, short text counts half: lengths alternate forever
        calls.append(len(s))
        return 2 * target if len(s) >= first else target // 2

    out = C._trim_to_tokens(raw, target, flip)
    assert len(calls) == 2 and len(set(calls)) == 2          # stopped on the repeat, not after 8
    assert len(out) == min(calls)                              # the best of the two (50% vs 100% off)
    assert _old_trim(raw, target, flip) == out
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "[filler trim]" in ln]
    assert len(lines) == 1 and "WARNING" in lines[0]


@pytest.mark.parametrize("target", [37, 150, 512, 1500, 4000])
@pytest.mark.parametrize("seed", [1, 42, 20260905 * 31 + 7])
def test_same_bytes_as_the_old_trim(target, seed):
    for variant, builder in (("F-NUM", C._build_filler_num), ("F-PROSE", C._build_filler_prose)):
        old_trim, C._trim_to_tokens = C._trim_to_tokens, lambda raw, t, fn, label="filler", max_iter=8: _old_trim(
            raw, t, fn, max_iter)
        try:
            want = builder(target, seed, _words)
        finally:
            C._trim_to_tokens = old_trim
        assert C.build_filler(target, seed=seed, count_fn=_words, variant=variant) == want


def test_each_filler_is_trimmed_once(capsys):
    n = []

    def counting(s):
        n.append(1)
        return _words(s)

    a = C.build_filler(150, seed=7, count_fn=counting)
    k = len(n)
    b = C.build_filler(150, seed=7, count_fn=counting)
    assert a == b and k > 0 and len(n) == k
    assert C.build_filler(150, seed=8, count_fn=counting) != a and len(n) > k
    assert capsys.readouterr().out.count("[filler trim]") == 2


def test_8192_session_fillers_converge_in_two_iterations(capsys):
    """The B3 v2 8192 session's per-turn fillers (150 tokens, seed 20260905 * 31 + turn) with the evalset counter."""
    import t2s_r2_session_growth as r2
    for idx in range(1, 31):
        C.build_filler(150, seed=20260905 * 31 + idx, count_fn=r2._approx_token_count)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "[filler trim]" in ln]
    assert len(lines) == 30 and all("WARNING" not in ln and "iters)" in ln for ln in lines)
    assert all(int(ln.split("(")[1].split()[0]) <= 2 for ln in lines)
