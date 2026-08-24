"""Synthetic filler context builder.

Generates semantically inert prose to reach a target token depth without
influencing probe answers. Three variants are supported:

  F-NUM (default) — numbered administrative log entries. Each entry embeds
      the same integer in six positions (entry number, compliance period,
      district, cycle, segment, reference). Used as the control filler in
      Stage C and the filler composition experiment.

  F-PROSE — neutral descriptive prose with no digits and no numerals in any
      form (no spelled-out numbers, ordinals, or identifiers). Subject matter
      is mundane landscape observation, unrelated to all probe domains. Digit
      freedom is verified programmatically before the filler is returned.

  F-STRUCT-NONNUM — structured entries with the same visual shape as F-NUM
      (consistent line format, repeated key positions) but with non-numeric
      word-based identifiers in place of integers. Separates structure from
      numerals as a potential fabrication driver.

Filler mode:
  unlabelled (default) — filler is prepended with no framing, banner, or
      instruction referring to it. The model receives filler as though it were
      ordinary accumulated context, which is the condition the degradation study
      intends to measure. Labelled filler would instead measure whether the
      model follows an instruction to ignore a block, conflating instruction-
      following with context degradation.
  labelled — filler is wrapped in <background_context> tags with an explicit
      "this is irrelevant" instruction. Retained as a comparison arm only.

All callers must pass filler_mode explicitly or accept the default (unlabelled).
"""

from __future__ import annotations

import random
import re
from typing import Callable

FILLER_MODES = ("unlabelled", "labelled")
DEFAULT_FILLER_MODE = "unlabelled"

# Empirically calibrated from smoke-test data (2026-08-13):
# At target=4096, api/chat prompt_eval_count yielded 3256 filler tokens from
# 16384 chars (4*4096), giving 16384/3256 = 5.03 chars/token for this template.
# The old value of 4.0 caused a 20% undershoot.
_CHARS_PER_TOKEN = 5.03

# ─── F-NUM ────────────────────────────────────────────────────────────────────

_TEMPLATE = (
    "Administrative log entry {n}: The oversight committee reviewed all "
    "submitted documentation for compliance period {n} and confirmed that "
    "operational metrics remained within established baseline parameters. "
    "No anomalies were recorded in district {n} during the reference interval. "
    "Budget allocations for cycle {n} were processed according to standing "
    "procedure without escalation. Routine maintenance of infrastructure "
    "segment {n} was completed on schedule and filed under reference {n}. "
)

# ─── F-PROSE ──────────────────────────────────────────────────────────────────

# Sentences contain no digits, no spelled-out numbers (zero through trillion),
# no ordinals (first, second, ...), no quantity words (pair, dozen, trio, etc.),
# and no content from probe domains (sensors, policies, regions, logs, config).
_PROSE_SENTENCES = [
    "The terrain beyond the ridge appeared uniform and unremarkable, with low vegetation occupying the flat ground between shallow depressions.",
    "Cloud cover moved steadily from the west, creating diffuse light that cast no distinct shadows across the open ground.",
    "The surface was pale and packed, showing little evidence of recent disturbance.",
    "A line of sparse shrubs marked the boundary between the open area and the wooded margin.",
    "The air carried a faint quality of dampness without being uncomfortable.",
    "Along the far edge of the plain, the ground level dropped gradually toward a dry streambed.",
    "The streambed showed exposed rock at the edges, smooth and light grey in colour.",
    "No water was present in the channel.",
    "The surrounding vegetation was dry and sparse, with foliage appearing bleached by prolonged sun exposure.",
    "Wind moved through the area in irregular gusts, shifting direction without pattern.",
    "The sky overhead was an undifferentiated pale grey, with no visible sun position detectable through the cloud layer.",
    "Ground temperature was moderate and the air in sheltered spots was still.",
    "The soil beneath the sparse ground cover was loose and dusty when disturbed.",
    "Bark on nearby trees showed texture consistent with mature growth, deeply furrowed and pale at the ridges.",
    "Foliage at the canopy level was dense but uniform, with no flowering visible.",
    "The margin between the open ground and the treeline was abrupt, with little transitional vegetation.",
    "Stone outcroppings along the western slope were weathered and largely obscured by lichen.",
    "The path through the lower area was compacted and clear of obstruction.",
    "Fallen material had accumulated at the base of the slope, mostly dry and light in colour.",
    "The horizon in every direction was flat or gently undulating, with no notable landmarks.",
    "Occasional movement in the upper canopy suggested wind that was not detectable at ground level.",
    "The colour of the grass varied from pale yellow near exposed areas to dull green in sheltered patches.",
    "Moisture in the soil increased noticeably near the depression, though standing water was absent.",
    "Clouds moved in from the west at a steady pace, thickening as the light declined.",
    "The texture of the soil changed from sandy to clayey within a short horizontal distance.",
    "Root systems from the larger trees had pushed through the surface in shaded areas.",
    "The underside of leaves showed a lighter colour than the upper surface, consistent with the species.",
    "Late afternoon light, when it appeared through gaps in the cloud cover, was warm and nearly horizontal.",
    "The edge of the woodland offered partial shelter from wind but increased humidity.",
    "Small stones were scattered across the slope, angular and consistent in colour with the local geology.",
    "Lichen growth on exposed rock faces was dense and varied in tone from pale grey to deep orange.",
    "The canopy closed completely in the interior of the woodland, reducing ground light to a dim ambient level.",
    "Leaf litter on the woodland floor was deep and moist near the centre, shallower and brittle at the margin.",
    "The bark of the dominant tree species was smooth when young and became deeply ridged with age.",
    "A shallow dry channel crossed the plain at an angle, its bed smooth and covered with fine pale material.",
    "The visible rock strata along the exposed face showed clear horizontal banding in tones of grey and buff.",
    "Grasses in the sheltered hollow were taller and greener than those on the exposed slope above.",
    "The breeze carried no identifiable scent across the open ground.",
    "Shadows lengthened across the terrain as the cloud layer thickened and light levels dropped.",
    "The soil at the base of the escarpment was dark and compacted, different in character from the slope above.",
    "Moss covered the lower portions of the exposed boulders, fading to bare rock above the moisture line.",
    "A persistent low haze softened the distant skyline without obscuring it entirely.",
    "The grass stems were dry and hollow, rustling against each other in the lightest movement of air.",
    "Below the escarpment the terrain flattened into a broad open area with poor drainage.",
    "The surface of the pond, where visible through the vegetation, reflected the pale sky without distortion.",
    "Reeds at the margin of the pond were upright and dense, showing no sign of insect or animal activity.",
    "The track along the base of the escarpment was partially overgrown but still navigable.",
    "Clay deposits near the streambed showed cracking patterns consistent with repeated wetting and drying.",
    "The upper branches of the tallest trees were visible above the general canopy level, bare and angular.",
    "Soil temperature at the surface was noticeably warmer than the air above it in the sheltered hollow.",
]

# Regex detecting any digit (ASCII), spelled-out integers, ordinals, or
# explicit quantity words. Used by assert_prose_clean().
_PROSE_FORBIDDEN = re.compile(
    r"""
    [0-9]                                           # any digit
    | \b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten
          |eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen
          |eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy
          |eighty|ninety|hundred|thousand|million|billion|trillion)\b
    | \b(?:first|second|third|fourth|fifth|sixth|seventh|eighth
          |ninth|tenth|eleventh|twelfth)\b
    | \b(?:pair|couple|dozen|trio|quartet|quintet|score)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def assert_prose_clean(text: str) -> None:
    """Raise AssertionError if text contains any digit or number word."""
    m = _PROSE_FORBIDDEN.search(text)
    if m:
        ctx = text[max(0, m.start() - 20): m.end() + 20]
        raise AssertionError(
            f"F-PROSE filler contains forbidden content {m.group()!r} at "
            f"position {m.start()}: ...{ctx!r}..."
        )


# ─── F-STRUCT-NONNUM ──────────────────────────────────────────────────────────

# Two word lists whose cross-product gives unique identifiers for each entry.
# No numeric content; visual shape mirrors F-NUM (repeated placeholder in
# consistent positions across a multi-sentence entry block).
_SNN_COLORS = [
    "COBALT", "SLATE", "AMBER", "BRONZE", "GRANITE", "IVORY", "ONYX",
    "CEDAR", "UMBER", "OCHRE", "TEAL", "MAUVE", "FLINT", "BIRCH", "CHALK",
    "AGATE", "BASALT", "DUNE", "FERN", "MOSS", "CLAY", "SHALE", "MIST",
    "RUST", "ASH",
]
_SNN_PLACES = [
    "RIDGE", "VALE", "CREST", "SHELF", "BLUFF", "HEATH", "DOWNS", "FORD",
    "GLEN", "MERE", "MOOR", "CAPE", "DELL", "KNOLL", "WICK", "HOLM",
    "BRAE", "BURN", "FELL", "HOLT", "SHAW", "BECK", "LOCH", "TARN", "FEN",
]

# Same sentence skeleton as F-NUM but {n} is replaced by {label}.
_SNN_TEMPLATE = (
    "Administrative log entry [{label}]: The oversight committee reviewed all "
    "submitted documentation for compliance period [{label}] and confirmed that "
    "operational metrics remained within established baseline parameters. "
    "No anomalies were recorded in district [{label}] during the reference interval. "
    "Budget allocations for cycle [{label}] were processed according to standing "
    "procedure without escalation. Routine maintenance of infrastructure "
    "segment [{label}] was completed on schedule and filed under reference [{label}]. "
)


def _trim_to_tokens(
    raw: str,
    target_tokens: int,
    count_fn: Callable[[str], int] | None,
    label: str = "filler",
    max_iter: int = 8,
) -> str:
    """Trim raw string to approximately target_tokens using count_fn refinement.

    Logs each iteration. Accepts the closest result if max_iter is exhausted
    without converging — never hangs.
    """
    target_chars = min(int(target_tokens * _CHARS_PER_TOKEN), len(raw))
    filler = raw[:target_chars].strip()
    if count_fn is None:
        return filler
    best = filler
    best_err = float("inf")
    for i in range(max_iter):
        actual = count_fn(filler)
        if actual <= 0:
            break
        err = abs(actual - target_tokens) / max(target_tokens, 1)
        is_best = err < best_err
        if is_best:
            best, best_err = filler, err
        print(f"  [{label} trim {i+1}/{max_iter}] {actual} tok  err={err:.1%}{' *' if is_best else ''}")
        if err <= 0.02:
            return filler
        new_chars = min(int(len(filler) * target_tokens / actual), len(raw))
        if new_chars == len(filler):
            break
        filler = raw[:new_chars].strip()
    if best_err > 0.02:
        print(f"  [{label}] WARNING: did not converge in {max_iter} iters, accepting best (err={best_err:.1%})")
    return best


def build_filler(
    target_tokens: int,
    seed: int = 42,
    count_fn: Callable[[str], int] | None = None,
    variant: str = "F-NUM",
) -> str:
    """Return filler text targeting target_tokens.

    variant:
      "F-NUM"           — numbered administrative log entries (default, control).
      "F-PROSE"         — digit-free descriptive landscape prose. Asserts clean
                          before returning; raises AssertionError if violated.
      "F-STRUCT-NONNUM" — same structure as F-NUM but word-based identifiers
                          instead of integers.

    If count_fn is provided the result is iteratively adjusted until within 2%
    of target_tokens or until 4 iterations are exhausted.
    """
    if target_tokens <= 0:
        return ""
    if variant == "F-NUM":
        return _build_filler_num(target_tokens, seed, count_fn)
    if variant == "F-PROSE":
        return _build_filler_prose(target_tokens, seed, count_fn)
    if variant == "F-STRUCT-NONNUM":
        return _build_filler_struct_nonnum(target_tokens, seed, count_fn)
    raise ValueError(f"Unknown filler variant: {variant!r}")


def _build_filler_num(
    target_tokens: int,
    seed: int,
    count_fn: Callable[[str], int] | None,
) -> str:
    rng = random.Random(seed)
    start = rng.randint(10_000, 99_999)
    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    chunks: list[str] = []
    total = 0
    i = start
    while total < needed_chars:
        chunk = _TEMPLATE.format(n=i)
        chunks.append(chunk)
        total += len(chunk)
        i += 1
    raw = "".join(chunks)
    return _trim_to_tokens(raw, target_tokens, count_fn)


def _build_filler_prose(
    target_tokens: int,
    seed: int,
    count_fn: Callable[[str], int] | None,
) -> str:
    rng = random.Random(seed)
    sentences = list(_PROSE_SENTENCES)
    rng.shuffle(sentences)
    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    chunks: list[str] = []
    total = 0
    idx = 0
    while total < needed_chars:
        sent = sentences[idx % len(sentences)]
        chunks.append(sent + " ")
        total += len(sent) + 1
        idx += 1
    raw = "".join(chunks)
    filler = _trim_to_tokens(raw, target_tokens, count_fn)
    assert_prose_clean(filler)
    return filler


def _build_filler_struct_nonnum(
    target_tokens: int,
    seed: int,
    count_fn: Callable[[str], int] | None,
) -> str:
    # Generate label sequence from cross-product of color × place names.
    labels = [f"{c}-{p}" for c in _SNN_COLORS for p in _SNN_PLACES]
    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    chunks: list[str] = []
    total = 0
    idx = 0
    while total < needed_chars:
        label = labels[idx % len(labels)]
        chunk = _SNN_TEMPLATE.format(label=label)
        chunks.append(chunk)
        total += len(chunk)
        idx += 1
    raw = "".join(chunks)
    return _trim_to_tokens(raw, target_tokens, count_fn)


def wrap_prompt(
    filler: str,
    probe_prompt: str,
    filler_mode: str = DEFAULT_FILLER_MODE,
) -> str:
    """Return the full prompt for a given filler and mode.

    unlabelled: filler + blank line + probe_prompt. No framing.
    labelled:   filler in <background_context> tags with an ignore instruction.
    """
    if filler_mode not in FILLER_MODES:
        raise ValueError(f"filler_mode must be one of {FILLER_MODES}, got {filler_mode!r}")
    if not filler:
        return probe_prompt
    if filler_mode == "unlabelled":
        return f"{filler}\n\n{probe_prompt}"
    # labelled
    return (
        "<background_context>\n"
        f"{filler}\n"
        "</background_context>\n\n"
        "The background context above is administrative filler unrelated to the "
        "question below. Answer using only your own knowledge.\n\n"
        f"{probe_prompt}"
    )
