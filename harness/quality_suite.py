"""RULER-style long-context quality probes, generated locally at any target prompt length.

Companion to harness/t2s_lab.py and harness/t2s_overnight.py: a night2-style phase calls
build_task()/run_task()/run_suite() the same way it already calls ov.do_call/ov.probe_sequence,
against a duck-typed Server (needs .tokenize(text) -> int and .chat(prompt, max_tokens, ignore_eos)
-> dict with at least outcome/output, matching harness/t2s_lab.py's Server.chat). This module does
not talk to a server itself and is not an orchestrator: it only builds prompts, scores outputs, and
returns plain-dict rows a caller's own lab.row(...)/lab.emit(...) can carry forward.

Task types (built locally, deterministic given (task_type, target_tokens, seed), so a row only ever
needs to record those three fields plus the metadata below to reconstruct the exact prompt again):

  niah_multikey            3 distinct key/value needles scattered in filler; ask for 1 named key's
                            value. Exact-match.
  niah_multivalue          1 key with 4 associated values scattered in filler; ask for all 4.
                            Set-F1 (order-independent).
  variable_tracking        A 5-step variable reassignment chain (X1 = n, X2 = X1, ...) scattered in
                            filler; ask for the final value. Exact-match.
  common_words_extraction  A word list with a known, strictly-decreasing frequency distribution; ask
                            for the top 10 most frequent words. Set-F1 (order-independent).
  system_rule_compliance   A fixed formatting rule at the very front of the prompt, a task after it.
                            Binary score: valid JSON with exactly the two required keys, content not
                            checked.
  tool_call_correctness    A fixed function schema at the front of the prompt, a natural-language
                            request at the end. Score requires valid JSON AND an exact argument match
                            against the one correct call for that request.

art_probe rows are not built here: they reuse evaluation/probes/segments.jsonl and scorers.py
exactly like harness/t2s_lab.py's load_probes() does (read-only; nothing in this module writes into
evaluation/probes/), loaded locally rather than by importing t2s_lab so this module does not pull in
t2s_lab's Windows-telemetry dependency chain (gguf_meta, level_zero_sysman, server_guard) just to
reuse one loader function.

Filler text for the RULER tasks reuses harness/context.py's build_filler(target_tokens, seed) in its
plain (no count_fn) mode: build_task() takes no server, so token counts are the same char-per-token
approximation build_filler already uses without a tokenizer. run_suite() has a real server and passes
its .tokenize as count_fn for the art-probe filler, same as t2s_lab.build_probe_prompts does.
"""

from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import context as ctx_mod

# Relative resolution matches a local dev checkout (this file at <repo>/harness/quality_suite.py, probes at
# <repo>/evaluation/probes). It does NOT match a deployed evo-t2s/evo-x2 machine: scripts/deploy_evo.py copies files
# flat into the deploy dir (C:\apu\ovn), with no evaluation/ alongside them -- the probes live in a separate full
# checkout at C:\apu\APU\evaluation\probes there (see t2s_lab.PROBES_DIR). Fall back to that fixed path when the
# relative one does not exist, so this module resolves correctly in both places without importing t2s_lab (and
# pulling in its Windows-only telemetry chain) just for this one constant.
_LOCAL_PROBES_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "probes"
_DEPLOYED_PROBES_DIR = Path(r"C:\apu\APU\evaluation\probes")
PROBES_DIR = _LOCAL_PROBES_DIR if _LOCAL_PROBES_DIR.exists() else _DEPLOYED_PROBES_DIR

TASK_TYPES = (
    "niah_multikey",
    "niah_multivalue",
    "variable_tracking",
    "common_words_extraction",
    "system_rule_compliance",
    "tool_call_correctness",
)


@dataclass
class Task:
    """One generated probe. prompt/expected are the (prompt, expected) pair the scorer needs;
    meta carries enough to reconstruct the prompt from build_task(task_type, target_tokens, seed)
    again, plus whatever a human wants for debugging (chosen needle, exact counts, etc.)."""
    task_type: str
    prompt: str
    expected: object
    scorer: str
    max_tokens: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- shared helpers

def _rand_token(rng: random.Random, length: int, alphabet: str = string.ascii_uppercase + string.digits) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def _scatter(filler: str, insertions: list[str], rng: random.Random) -> str:
    """Interleave insertions into filler at positions spread across it, in the given order (order
    matters for variable_tracking's causal chain; for the niah tasks the insertions are already
    independent of each other so order is only cosmetic).

    filler is split on '. ' into pseudo-sentences and insertions are dropped in after evenly spaced
    spans. If filler has fewer spans than insertions, everything is appended at the end instead; this
    only happens at target_tokens small enough that the filler itself is a couple of sentences, which
    is below any depth this suite is meant to run at."""
    spans = [s for s in filler.split(". ") if s.strip()]
    n = len(insertions)
    if n == 0:
        return filler
    if len(spans) <= n:
        return filler.rstrip(". ") + ". " + " ".join(insertions)
    step = max(len(spans) // (n + 1), 1)
    out: list[str] = []
    next_ins = 0
    for i, s in enumerate(spans):
        out.append(s)
        if next_ins < n and i > 0 and (i + 1) % step == 0:
            out.append(insertions[next_ins])
            next_ins += 1
    while next_ins < n:
        out.append(insertions[next_ins])
        next_ins += 1
    return ". ".join(out) + "."


def truncate_front(text: str, fraction: float = 0.5) -> str:
    """Drop the first `fraction` of text's characters. Used by the truncation control: dropping the
    front of a prompt that carries a needle near the front should lower a task's score."""
    return text[int(len(text) * fraction):]


_PUNCT_STRIP = str.maketrans("", "", string.punctuation)


def normalize_token(s: str) -> str:
    return s.strip().translate(_PUNCT_STRIP).lower()


_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)


def _strip_fences(text: str) -> str:
    m = _FENCE.search(text or "")
    return m.group(1) if m else (text or "")


def _extract_json_object(text: str) -> str | None:
    """First fenced block if present, else the outermost {...} span. Same approach as
    evaluation/probes/scorers.py's score_schema, reimplemented locally so this module has no import
    dependency on evaluation/probes/ beyond the read-only scorer/segment loading in load_probes()."""
    raw = _strip_fences(text).strip()
    if not raw.startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        if i == -1 or j == -1 or j < i:
            return None
        raw = raw[i:j + 1]
    return raw


# ---------------------------------------------------------------- scorers

def score_exact_local(output: str, expected: str) -> tuple[float, str]:
    """Exact-match, tolerant of a short preamble: only the last non-empty line is compared, then a
    substring fallback (expected value appears anywhere in the normalized output) catches a model
    that answers correctly but does not put the value on its own line."""
    lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
    got_raw = lines[-1] if lines else (output or "")
    got = normalize_token(got_raw)
    want = normalize_token(str(expected))
    if got == want:
        return 1.0, "exact match"
    if want and want in normalize_token(output or ""):
        return 1.0, "substring match"
    return 0.0, f"got={got_raw!r} want={expected!r}"


def parse_list_output(text: str) -> set[str]:
    """Split a comma/newline/semicolon/'and'-separated list into a normalized set of items."""
    if not text:
        return set()
    parts = re.split(r"[,\n;]+|\band\b", text)
    return {normalize_token(p) for p in parts if normalize_token(p)}


def score_set_f1(output: str, expected) -> tuple[float, str]:
    """Order-independent set-F1. expected may be any iterable of strings (a set, list, ...)."""
    got = parse_list_output(output)
    want = {normalize_token(x) for x in expected}
    if not want and not got:
        return 1.0, "both empty"
    inter = got & want
    precision = len(inter) / len(got) if got else 0.0
    recall = len(inter) / len(want) if want else 0.0
    if precision + recall == 0:
        return 0.0, f"got={sorted(got)} want={sorted(want)}"
    f1 = 2 * precision * recall / (precision + recall)
    return f1, f"precision={precision:.2f} recall={recall:.2f} got={sorted(got)} want={sorted(want)}"


def score_json_keys(output: str, expected_keys) -> tuple[float, str]:
    """Binary: valid JSON object whose key set is exactly expected_keys. Values are not checked."""
    raw = _extract_json_object(output)
    if raw is None:
        return 0.0, "no JSON object found"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return 0.0, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return 0.0, "JSON value is not an object"
    if set(obj.keys()) == set(expected_keys):
        return 1.0, "valid, exact key set"
    return 0.0, f"keys={sorted(obj.keys())} want={sorted(expected_keys)}"


def score_tool_call(output: str, expected_call: dict) -> tuple[float, str]:
    """Binary: valid JSON AND an exact match (name + every argument, including list order) against
    the one known-correct call. No partial credit: a tool call with a wrong argument is as wrong as
    one that fails to parse, since either would misfire if actually executed."""
    raw = _extract_json_object(output)
    if raw is None:
        return 0.0, "no JSON object found"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return 0.0, f"invalid JSON: {e}"
    if obj == expected_call:
        return 1.0, "exact call match"
    return 0.0, f"got={obj!r} want={expected_call!r}"


_SCORERS: dict[str, Callable[[str, object], tuple[float, str]]] = {
    "exact": score_exact_local,
    "set_f1": score_set_f1,
    "json_keys": score_json_keys,
    "tool_call": score_tool_call,
}


def score_task(scorer_name: str, output: str, expected) -> tuple[float, str]:
    """Public entry point for a caller that does its own chat call (e.g. one driving a different API, like Ollama's,
    rather than the srv.tokenize/.chat shape run_task expects) and only needs the scoring half. scorer_name is a
    Task's .scorer field (one of _SCORERS' keys, not the task_type); an unknown name raises KeyError with the valid
    set, rather than silently mis-scoring."""
    if scorer_name not in _SCORERS:
        raise KeyError(f"unknown scorer {scorer_name!r}, expected one of {sorted(_SCORERS)}")
    return _SCORERS[scorer_name](output, expected)


# ---------------------------------------------------------------- fabrication vs refusal

_REFUSAL_PATTERNS = re.compile(
    r"\b("
    r"i\s+don'?t\s+know|i\s+do\s+not\s+know|"
    r"can(?:no|')?t\s+find|cannot\s+find|"
    r"not\s+(?:provided|mentioned|stated|specified|given|available)|"
    r"no\s+(?:information|data|mention)|"
    r"does\s+not\s+(?:mention|contain|provide|specify)|"
    r"cannot\s+determine|unable\s+to\s+determine|"
    r"insufficient\s+information|"
    r"i'?m\s+not\s+sure|i\s+am\s+not\s+sure|"
    r"impossible\s+to\s+(?:determine|answer)"
    r")\b",
    re.IGNORECASE,
)


def classify_fabrication_or_refusal(output: str | None) -> str:
    """Classify a wrong (score < 1.0) output as 'refusal' or 'fabrication'.

    Heuristic: a fixed set of refusal/inability phrases (regex, case-insensitive) is searched for
    anywhere in the output. A hit is 'refusal'; anything else scored wrong, including empty output,
    is 'fabrication' unless the output is empty (empty output is treated as a refusal: the model
    produced nothing rather than inventing something).

    LIMITATIONS (read before trusting this on a real run):
      - Hedge-then-answer is not detected. "I don't know, but it's probably 12" is classified
        'refusal' even though a specific value follows; evaluation/probes/scorers.py's
        classify_abstention() implements the fuller hedge/pivot check for the art probes
        specifically, and is not reused here because its phrase list and hedge regex were tuned to
        that probe set's wording, not the RULER-style tasks built in this module.
      - A model that fabricates using refusal-adjacent language ("the passage is unclear, so it must
        be X") can be misclassified as refusal if the phrase match lands before the fabricated
        value.
      - Non-English refusals, or refusals phrased without any of the listed keywords, fall through to
        'fabrication'.
      - This is a cheap triage signal for eyeballing the mix of failure modes across a run, not a
        certified per-row label; do not report it as ground truth without a manual spot check.
    """
    if not output or not output.strip():
        return "refusal"
    return "refusal" if _REFUSAL_PATTERNS.search(output) else "fabrication"


# ---------------------------------------------------------------- niah_multikey / niah_multivalue

def _build_niah_multikey(target_tokens: int, seed: int, n_keys: int = 3, count_fn=None) -> Task:
    rng = random.Random(seed)
    keys = [f"KEY-{_rand_token(rng, 4)}" for _ in range(n_keys)]
    values = [_rand_token(rng, 6, string.digits) for _ in range(n_keys)]
    sentences = [f"The registration code associated with {k} is {v}." for k, v in zip(keys, values)]
    fill_target = max(target_tokens - 25 * n_keys - 40, 64)
    filler = ctx_mod.build_filler(fill_target, seed=seed, count_fn=count_fn)
    body = _scatter(filler, sentences, rng)
    chosen_idx = rng.randrange(n_keys)
    chosen_key, chosen_val = keys[chosen_idx], values[chosen_idx]
    question = (f"\n\nQuestion: What is the registration code associated with {chosen_key}? "
                "Respond with ONLY the code, no explanation.")
    meta = {"keys": keys, "values": values, "chosen_key": chosen_key, "n_keys": n_keys}
    return Task("niah_multikey", body + question, chosen_val, "exact", 32, meta)


def _build_niah_multivalue(target_tokens: int, seed: int, n_values: int = 4, count_fn=None) -> Task:
    rng = random.Random(seed)
    key = f"PROJECT-{_rand_token(rng, 4)}"
    values = [_rand_token(rng, 5, string.ascii_uppercase + string.digits) for _ in range(n_values)]
    sentences = [f"Project {key} was assigned milestone tag {v}." for v in values]
    fill_target = max(target_tokens - 20 * n_values - 60, 64)
    filler = ctx_mod.build_filler(fill_target, seed=seed, count_fn=count_fn)
    body = _scatter(filler, sentences, rng)
    question = (f"\n\nQuestion: List every milestone tag recorded for project {key}. "
                "Respond with ONLY the tags, comma-separated, in any order.")
    meta = {"key": key, "values": values, "n_values": n_values}
    return Task("niah_multivalue", body + question, set(values), "set_f1", 96, meta)


# ---------------------------------------------------------------- variable_tracking

def _build_variable_tracking(target_tokens: int, seed: int, n_steps: int = 5, count_fn=None) -> Task:
    rng = random.Random(seed)
    final_value = rng.randint(1, 99)
    var_names = [f"X{i + 1}" for i in range(n_steps)]
    lines = [f"{var_names[0]} = {final_value}."]
    for i in range(1, n_steps):
        lines.append(f"{var_names[i]} = {var_names[i - 1]}.")
    fill_target = max(target_tokens - 10 * n_steps - 40, 64)
    filler = ctx_mod.build_filler(fill_target, seed=seed, count_fn=count_fn)
    body = _scatter(filler, lines, rng)
    last_var = var_names[-1]
    question = f"\n\nQuestion: What is the final numeric value of {last_var}? Respond with ONLY the number."
    meta = {"chain": list(zip(var_names, lines)), "final_value": final_value, "last_var": last_var}
    return Task("variable_tracking", body + question, str(final_value), "exact", 16, meta)


# ---------------------------------------------------------------- common_words_extraction

# 16 distinct, low-frequency-in-ordinary-English content words. Any strictly positive per-word
# scaling factor k gives strictly decreasing counts (n - i) * k, so there is no tie at the rank
# 10/11 boundary regardless of target_tokens; k just controls how many total repeats are needed.
_CW_VOCAB = ["beacon", "ledger", "orbit", "cipher", "harbor", "quartz", "meridian", "paddock",
             "thicket", "ember", "granary", "isthmus", "cobalt", "junction", "willow", "furrow"]

# Padding word pool for common_words_extraction, extracted from context._PROSE_SENTENCES (real English prose already
# vetted elsewhere in this codebase for build_filler's F-PROSE variant). Original design used synthetic per-position
# tokens like f"n{seed}{i:06d}" for padding, each appearing exactly once so none could compete with the designed
# vocabulary's counts -- correct in principle, but those strings are unfamiliar 10-character digit/letter mixes with
# no real-language structure, so a BPE tokenizer fragments them far more than context.py's calibrated 5.03
# chars/token assumes (confirmed live on evo-x2, 2026-09-29: a 2000-token-intended prompt actually tokenized to
# 12,657 tokens, blowing past the 8192-token context and failing every call outright). Real English words from an
# already-used prose bank tokenize close to that same 5.03 chars/token calibration, fixing the length blowup; see
# _pad_words_for below for how repeats (unavoidable once padding needs exceed the pool size) are kept safely below
# the vocabulary's own minimum count instead of relying on per-position uniqueness.
_PAD_WORD_POOL = sorted({w for s in ctx_mod._PROSE_SENTENCES for w in re.findall(r"[a-z]{3,10}", s.lower())}
                        - set(_CW_VOCAB))

# _PAD_WORD_POOL's words average ~6.4 characters (plus a separating space = ~7.4) versus context.py's
# _CHARS_PER_TOKEN=5.03 calibration (measured on the F-NUM template's shorter, function-word-heavy text), i.e. about
# 1.47 tokens per pool word rather than 1. Budgeting this module's word counts directly against target_tokens (as if
# 1 word = 1 token) is what caused the original padding-token bug's replacement fix to still overshoot by about that
# same 1.47x; dividing the word budget by this ratio corrects for it.
_EST_TOKENS_PER_PAD_WORD = 7.4 / ctx_mod._CHARS_PER_TOKEN


def _pad_words_for(pad_needed: int, rng: random.Random) -> list[str]:
    """pad_needed words from _PAD_WORD_POOL, cycling (not sampling with replacement) so every pool word's exact
    repeat count is either floor or ceil(pad_needed / pool_size) -- a hard deterministic bound, not a statistical
    average, so the caller can size the vocabulary's minimum count to safely exceed it at any target_tokens."""
    pool = list(_PAD_WORD_POOL)
    rng.shuffle(pool)
    reps = -(-pad_needed // len(pool))  # ceil division
    words = (pool * reps)[:pad_needed]
    rng.shuffle(words)
    return words


def _build_common_words(target_tokens: int, seed: int, n_top: int = 10, count_fn=None) -> Task:
    rng = random.Random(seed)
    vocab = list(_CW_VOCAB)
    rng.shuffle(vocab)
    n = len(vocab)
    top_words = vocab[:n_top]
    question = ("\n\nQuestion: Of the words above, which 10 appear most frequently? List exactly 10 "
                "words, comma-separated, in any order.")

    def _build_for_word_budget(word_budget, rng):
        # Budget in WORD units, not tokens: dividing by _EST_TOKENS_PER_PAD_WORD corrects for the pool's real
        # words costing more than 1 token each (see that constant's comment). pad_needed and the resulting max
        # pool-word repeat count (see _pad_words_for) both scale with the word budget, so k must too:
        # max_pad_repeat is computed first and k set to clear it with a safety margin.
        approx_pad_needed = max(word_budget - 8, 0)
        max_pad_repeat = -(-approx_pad_needed // len(_PAD_WORD_POOL))
        k = max(1, word_budget // 400, max_pad_repeat + 2)
        counts = [(n - i) * k for i in range(n)]  # strictly decreasing by construction
        units: list[str] = []
        for w, c in zip(vocab, counts):
            units.extend([w] * c)
        pad_needed = max(word_budget - len(units) - 8, 0)
        units.extend(_pad_words_for(pad_needed, rng))
        rng.shuffle(units)
        passage = "Observed terms, in order: " + " ".join(units) + "."
        return passage, counts, k

    word_budget = max(int(target_tokens / _EST_TOKENS_PER_PAD_WORD) - 8, 0)
    # count_fn=None: identical to the pre-count_fn code path (same rng object, threaded through in the same order),
    # so every existing (task_type, target_tokens, seed) prompt stays byte-identical. count_fn given: iterative
    # refinement against a real tokenizer (mirroring context._trim_to_tokens's own pattern) reseeds a fresh rng
    # each attempt so the search is reproducible but does not need to match the no-count_fn text.
    if count_fn is None:
        passage, counts, k = _build_for_word_budget(word_budget, rng)
    else:
        for attempt in range(4):
            passage, counts, k = _build_for_word_budget(word_budget, random.Random(f"{seed}:{attempt}"))
            actual = count_fn(passage + question)
            if actual <= 0:
                break
            err = abs(actual - target_tokens) / max(target_tokens, 1)
            if err <= 0.02:
                break
            word_budget = max(int(word_budget * target_tokens / actual), 1)
    meta = {"counts": dict(zip(vocab, counts)), "top_words": top_words, "n_top": n_top, "pad_word_k_margin": k}
    return Task("common_words_extraction", passage + question, set(top_words), "set_f1", 96, meta)


# ---------------------------------------------------------------- system_rule_compliance

SYSTEM_RULE = ('Answer only in JSON with exactly two keys, "answer" and "source", and no other text '
               "before or after the JSON object.")


def _build_system_rule(target_tokens: int, seed: int, count_fn=None) -> Task:
    niah = _build_niah_multikey(target_tokens, seed, n_keys=1, count_fn=count_fn)
    prompt = SYSTEM_RULE + "\n\n" + niah.prompt
    meta = dict(niah.meta, rule=SYSTEM_RULE)
    return Task("system_rule_compliance", prompt, ["answer", "source"], "json_keys", 64, meta)


# ---------------------------------------------------------------- tool_call_correctness

TOOL_SCHEMA = {
    "name": "schedule_meeting",
    "description": "Schedule a meeting on the shared calendar.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "date": {"type": "string", "description": "ISO 8601 date, YYYY-MM-DD"},
            "time": {"type": "string", "description": "24-hour clock, HH:MM"},
            "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "date", "time", "duration_minutes", "attendees"],
    },
}

_MEETING_TITLES = ["Budget Review", "Roadmap Sync", "Vendor Check-in", "Design Critique", "Postmortem"]
_ATTENDEE_NAMES = ["alice", "bob", "carol", "dave", "erin"]


def _build_tool_call(target_tokens: int, seed: int, count_fn=None) -> Task:
    rng = random.Random(seed)
    title = rng.choice(_MEETING_TITLES)
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    time_ = f"{rng.randint(8, 17):02d}:{rng.choice(['00', '15', '30', '45'])}"
    duration = rng.choice([15, 30, 45, 60])
    attendees = [f"{name}@example.com" for name in rng.sample(_ATTENDEE_NAMES, rng.randint(1, 3))]
    request = (f"Schedule a meeting titled '{title}' on {date} at {time_} for {duration} minutes "
               f"with {', '.join(attendees)}.")
    schema_block = ("Function schema (call this function exactly once, with these exact argument "
                     "names and types):\n" + json.dumps(TOOL_SCHEMA, indent=2))
    fill_target = max(target_tokens - 300, 64)
    filler = ctx_mod.build_filler(fill_target, seed=seed, count_fn=count_fn)
    prompt = (schema_block + "\n\n" + filler + "\n\nRequest: " + request +
              "\n\nRespond with ONLY a JSON object of the form {\"name\": ..., \"arguments\": {...}}.")
    expected_call = {"name": TOOL_SCHEMA["name"],
                      "arguments": {"title": title, "date": date, "time": time_,
                                    "duration_minutes": duration, "attendees": attendees}}
    meta = {"request": request, "expected_call": expected_call}
    return Task("tool_call_correctness", prompt, expected_call, "tool_call", 128, meta)


TASK_BUILDERS: dict[str, Callable[..., Task]] = {
    "niah_multikey": lambda tt, s, count_fn=None: _build_niah_multikey(tt, s, n_keys=3, count_fn=count_fn),
    "niah_multivalue": lambda tt, s, count_fn=None: _build_niah_multivalue(tt, s, n_values=4, count_fn=count_fn),
    "variable_tracking": lambda tt, s, count_fn=None: _build_variable_tracking(tt, s, n_steps=5, count_fn=count_fn),
    "common_words_extraction": _build_common_words,
    "system_rule_compliance": _build_system_rule,
    "tool_call_correctness": _build_tool_call,
}


def build_task(task_type: str, target_tokens: int, seed: int = 42, count_fn=None) -> Task:
    """Build one Task deterministically. Same (task_type, target_tokens, seed) always reproduces the
    identical prompt/expected pair, so a result row never needs to store the (potentially huge)
    prompt text itself, only these three values plus Task.meta -- EXCEPT when count_fn is given, since a
    different tokenizer's real counts can change how much filler gets generated to hit target_tokens.

    count_fn: an optional real tokenizer callback (text -> int token count), the same shape context.py's own
    build_filler/_trim_to_tokens already accept. Without it, filler sizing uses context._CHARS_PER_TOKEN's single
    fixed estimate, which was calibrated against Qwen3's tokenizer (see context.py) and under-counts by about 18-20%
    against Llama 3.1/3.3's tokenizer (real, more BPE-efficient prose) -- see analysis/offline_token_calibration.py,
    which found this on 2026-09-29. Passing a real tokenizer's count_fn here (or the live server's own /tokenize,
    matching how build_probe_prompts already does it) makes every family calibrate correctly, since the iterative
    refinement in context._trim_to_tokens converges toward whatever count_fn actually reports, regardless of which
    tokenizer family that is."""
    try:
        builder = TASK_BUILDERS[task_type]
    except KeyError:
        raise ValueError(f"unknown task_type {task_type!r}; must be one of {sorted(TASK_BUILDERS)}")
    return builder(target_tokens, seed, count_fn=count_fn)


# ---------------------------------------------------------------- art probes (reused, read-only)

def load_probes():
    """Load the scorers module and the five art_* probes, exactly like harness/t2s_lab.py's
    load_probes(): the scorers module is imported from its file path rather than as a package, so
    this works regardless of whether evaluation/probes is on sys.path. PROBES_DIR is resolved
    relative to this file (repo_root/evaluation/probes) rather than t2s_lab.py's hardcoded deployment
    path, so this also works in a dev checkout. Read-only: nothing here writes into
    evaluation/probes/.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    segs = {}
    for line in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            segs[d["id"]] = d
    return mod, [segs[f"art_{i:02d}"] for i in range(1, 6)]


def build_probe_prompts(srv, fill_tokens: int, probes: list[dict], seed: int = 42):
    """One prompt per probe: filler, artifact, question (artifact adjacent to its question), each
    about fill_tokens tokens total. Same left-truncation approach as harness/t2s_lab.py's
    build_probe_prompts, duplicated here (not imported) so this module does not pull in t2s_lab's
    Windows-telemetry dependency chain just to reuse this one helper."""
    tk = srv.tokenize
    min_aq = min(tk(p["artifact"].strip()) + tk(p["question"].strip()) for p in probes)
    filler_full = ctx_mod.build_filler(max(fill_tokens - min_aq, 256) + 64, seed=seed, count_fn=tk)
    full_tokens = tk(filler_full)

    def left_trunc(text, target):
        if tk(text) <= target:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            if tk(text[mid:]) <= target:
                hi = mid
            else:
                lo = mid + 1
        return text[lo:]

    out = []
    for p in probes:
        a, q = p["artifact"].strip(), p["question"].strip()
        per = max(fill_tokens - tk(a) - tk(q), 64)
        filler = left_trunc(filler_full, min(per, full_tokens))
        prompt = f"{filler}\n\n{a}\n\n{q}"
        out.append((p, prompt, tk(prompt)))
    return out


def _run_art_probes(srv, target_tokens: int, seed: int, max_tokens: int = 32) -> list[dict]:
    scorers_mod, probes = load_probes()
    rows = []
    for probe, prompt, n_tok in build_probe_prompts(srv, target_tokens, probes, seed=seed):
        res = srv.chat(prompt, max_tokens, False)
        output = res.get("output") if res.get("outcome") == "ok" else None
        if output is not None:
            try:
                score, detail = scorers_mod.score(
                    {"id": probe["id"], "scorer_type": probe["scorer_type"], "expected": probe["expected"]},
                    output)
            except Exception as e:
                score, detail = None, f"scorer_error:{e}"
        else:
            score, detail = None, res.get("error")
        fab = classify_fabrication_or_refusal(output) if (output is not None and score is not None and score < 1.0) else None
        rows.append({
            "task_type": "art_probe", "probe_id": probe["id"], "target_tokens": target_tokens, "seed": seed,
            "prompt_tokens": n_tok, "outcome": res.get("outcome"), "score": score, "score_detail": detail,
            "fabrication_or_refusal": fab, "meta": {"probe_id": probe["id"], "scorer_type": probe["scorer_type"]},
        })
    return rows


# ---------------------------------------------------------------- running a task / the full suite

def run_task(srv, task: Task, target_tokens: int, seed: int) -> dict:
    """Run one Task against srv (duck-typed like harness/t2s_lab.py's Server: .tokenize(text) -> int,
    .chat(prompt, max_tokens, ignore_eos) -> dict with at least outcome/output) and return one row.

    ignore_eos is False, matching t2s_overnight.probe_sequence: these have short, well-defined
    answers, not open-ended generation, so the model should stop on its own well inside max_tokens.
    target_tokens/seed are recorded (not re-derived from task) because run_task also runs tasks
    built elsewhere, including ones mutated for a control (e.g. the front-truncation control uses
    dataclasses.replace() to swap in a shortened prompt without re-deriving it from build_task).
    """
    prompt_tokens = srv.tokenize(task.prompt)
    res = srv.chat(task.prompt, task.max_tokens, False)
    output = res.get("output") if res.get("outcome") == "ok" else None
    if output is None:
        score, detail = None, res.get("error", "no output")
    else:
        score, detail = _SCORERS[task.scorer](output, task.expected)
    fab = classify_fabrication_or_refusal(output) if (output is not None and score is not None and score < 1.0) else None
    return {
        "task_type": task.task_type, "target_tokens": target_tokens, "seed": seed,
        "prompt_tokens": prompt_tokens, "outcome": res.get("outcome"), "score": score,
        "score_detail": detail, "fabrication_or_refusal": fab, "meta": task.meta,
    }


def run_suite(srv, target_tokens: int, seed: int = 42, *, include_art_probes: bool = True) -> list[dict]:
    """Run one of each RULER-style task type (TASK_TYPES) plus, by default, the five art_* probes,
    all at the same target_tokens/seed, against srv. Returns a flat list of row dicts, one per task,
    in the order TASK_TYPES lists them followed by the art probes. Callers running this inside a
    night2-style phase pass each row straight to their own lab.row(...)/lab.emit(...); this function
    does not touch any lab object itself.
    """
    rows = [run_task(srv, build_task(tt, target_tokens, seed), target_tokens, seed) for tt in TASK_TYPES]
    if include_art_probes:
        rows.extend(_run_art_probes(srv, target_tokens, seed))
    return rows
