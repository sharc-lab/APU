# Workload pack

An SME-realistic workload test set with deterministic grading: every item
has a known-correct answer a script can check automatically (no
LLM-as-judge). It doubles as Paper 1's real-workload test set for the
envelope model and as the demo's input library.

Build scripts live in `scripts/`. Generated items live in `items/*.jsonl`
(one family per file, one JSON object per line). `grade.py` is the
deterministic grader and self-check. No model is called anywhere in this
pack; every item's oracle answer is checked against its own grading rule.

## CORRECTION (2026-10-01, controller, read this first)

The "honesty note" below is wrong, and for a specific, checkable reason: the subagent that wrote it worked in
a worktree branched from commit `5ddecb4`, roughly 140+ commits behind real main at the time, and never
fast-forward-merged before searching the repo (the same stale-checkout mistake several other subagents in
this session hit and corrected; this one did not). `harness/t2s_r2_session_growth.py`,
`analysis/agent_traces.py`, and `results/traces/agent_step_lengths.parquet` all exist on real main and were
committed well before this pack was built -- they were simply invisible from that stale branch point.

Consequence: family (d) (`d_r2_sessions.jsonl`) is new, independently-written session-generation code that
duplicates `harness/t2s_r2_session_growth.py`'s real `generate_session` function rather than reusing it, and
family (e) (`e_trace_mix.jsonl`) uses a synthetic lognormal length distribution instead of the real nebius-
rebench-openhands/SWE-Gym empirical CDF in `results/traces/agent_step_lengths.parquet`, which was available
the whole time. Both families are real, grade-checked, committed items -- they are not wrong, just not built
from the real generators/data this pack's own design called for. Rebuilding (d) against the real
`generate_session` and (e) against the real trace parquet is a real follow-up item, not done in this pass.

## Honesty note on prior art (read before trusting the task-spec framing, except where corrected above)

The build task this pack was written from described several pieces of
"existing prior art" in this repository to reuse: a Q0 / quality_suite
long-document generator in `harness/t2s_k1_ollama.py`, a multi-turn session
generator in `harness/t2s_r2_session_growth.py`, a tokenizer convention in
`analysis/agent_traces.py`, a real trace file at
`results/traces/agent_step_lengths.parquet`, and a "CWE" exclusion
convention documented somewhere in `docs/`.

Before writing any code we searched the whole repository (file names, `git
ls-files`, and full-text search for `qs_build_task`, `Q0`, `quality_suite`,
`generate_session`, `agent_traces`, `nebius`, `rebench`, `openhands`,
`swe-gym`/`swegym`, `*.parquet`, and `CWE`). **None of these exist anywhere
in this repository.** There is no long-document QA generator, no multi-turn
session generator, no `analysis/agent_traces.py`, no `results/traces/`
directory at all, and the string "CWE" does not appear anywhere in the repo
(so no exclusion rule was applied -- there was nothing to exclude by).

Given that, every family below is new code written for this pack, not reused
prior art. Family (e) in particular cannot be built the way the task
describes (sampling the real trace CDF), because the input file it depends
on does not exist; see family (e) below for what was built instead.

## Tokenizer

Prompt-token counts use `transformers.AutoTokenizer` with
`Qwen/Qwen2.5-7B-Instruct`'s real tokenizer (a real, reachable tokenizer in
the same model family -- Qwen3 -- as `configs/default.yaml`'s default
`qwen3:4b` model). This repo's own Python (3.12, see `pyproject.toml`) does
not have `transformers` or `datasets` installed and neither is a declared
dependency; the build scripts were run instead with the machine's separate
Python 3.8 install at `C:\Python38\python.exe`, which has `transformers
4.46.3` and `datasets 3.1.0` available. `grade.py` itself needs neither
package and runs fine under the repo's normal Python.

## Families and counts (400 items total)

| Family | Dir/file | Count | Design |
|---|---|---|---|
| (a) long-document QA | `items/a_longdoc_qa.jsonl` | 120 | Needle-in-haystack: synthetic SME-style filler document with a buried verification code; question asks for the code. New code (`scripts/build_a_longdoc_qa.py`); no prior-art generator exists in this repo (see above). |
| (b) function-calling / extraction | `items/b_function_calling.jsonl` | 80 | 60 function-call items (user request + 8 available function schemas, deterministic exact name+argument match) and 20 extraction items (free text to fixed-schema JSON). Self-contained design "inspired by" BFCL's item shape (request + tool schema + single correct call), not a literal BFCL replica. `scripts/build_b_function_calling.py`. |
| (c) GSM8K | `items/c_gsm8k.jsonl` | 80 | Real `openai/gsm8k` dataset (`main` config, `test` split), pulled live via `datasets.load_dataset(..., streaming=True)`. Oracle is the dataset's own final-answer field (text after `####`). `scripts/build_c_gsm8k.py`. |
| (d) multi-turn agent sessions | `items/d_r2_sessions.jsonl` | 60 | Simulated session transcripts: a standing tool-use rule, a fact stated once, several tool-requiring turns, then a recall question. Grading checks both tool-call-sequence compliance and exact recall. New code; no `t2s_r2_session_growth.py` exists in this repo. `scripts/build_d_r2_sessions.py`. |
| (e) context-length mix | `items/e_trace_mix.jsonl` | 60 | **Not** sampled from a real trace CDF -- `results/traces/agent_step_lengths.parquet` does not exist anywhere in this repo (verified by search, not assumed). Lengths are instead drawn from a clearly-labeled synthetic substitute lognormal distribution (median ~3000 tokens, heavy right tail to 120K), reusing family (a)'s needle-QA document generator. Every item is tagged `"length_distribution_source": "synthetic_substitute"`. `scripts/build_e_trace_mix.py`. |

120 + 80 + 80 + 60 + 60 = 400.

Split rationale: (a) gets the largest share because it is the family the
envelope model cares about most (prompt-length sweep from 2K-120K, 12
buckets x 10 items), (c) and (b) are capped near the size of a useful
few-dozen-item probe set for their task types, and (d)/(e) are smaller
because each item is itself a multi-step construct (a full session, or a
length drawn from a distribution rather than a fixed grid point).

## Grade-check results (real, from `grade.py`, not from memory)

```
TOTAL ITEMS: 400
  longdoc_qa: 120 items
  function_calling: 80 items
  gsm8k: 80 items
  r2_sessions: 60 items
  trace_length_mix: 60 items
GRADE-CHECK PASS: 400/400
```

Every item's own oracle answer scores exactly 1.0 against its own grading
rule. Zero exceptions; nothing was dropped or patched to fake a pass -- all
400 items passed on the first run of the final generator code.

## Length distribution (real, from `grade.py`)

Full pack (400 items), prompt tokens:

```
p50=356  p90=50642  p99=120010
min=50  max=120017  mean=13834
```

Per family:

```
longdoc_qa:        p50=28006  p90=96012  p99=120014  (2K-120K design range)
function_calling:  p50=347    p90=363    p99=367
gsm8k:              p50=80     p90=117    p99=150
r2_sessions:        p50=177    p90=205    p99=209
trace_length_mix:   p50=3684   p90=14262  p99=38498
```

**Comparison to the real trace CDF: not possible.**
`results/traces/agent_step_lengths.parquet` does not exist in this
repository, so there is no real nebius-rebench-openhands / SWE-Gym empirical
CDF to compare against. The numbers above for `trace_length_mix` are from the
synthetic substitute distribution described in family (e), not a comparison
to a real trace. If a real trace file is added to the repo later, family (e)
should be rebuilt from it and this section updated with the true comparison.

## Licenses

- **(c) GSM8K**: real data from the Hugging Face dataset `openai/gsm8k`
  (`main` config), license **MIT** per the dataset's own Hub metadata
  (`license:mit` tag, confirmed via the Hugging Face Hub API at build time,
  not from memory). Citation: Cobbe et al., "Training Verifiers to Solve Math
  Word Problems" (arXiv:2110.14168).
- **(a), (b), (d), (e)**: fully synthetic, generated by the scripts in this
  directory from fixed sentence/template pools and a seeded RNG. No external
  data, no license needed.

## Reproducing

```
# Build (needs transformers + datasets; use the Python 3.8 install that has them):
/c/Python38/python.exe scripts/build_a_longdoc_qa.py
/c/Python38/python.exe scripts/build_b_function_calling.py
/c/Python38/python.exe scripts/build_c_gsm8k.py
/c/Python38/python.exe scripts/build_d_r2_sessions.py
/c/Python38/python.exe scripts/build_e_trace_mix.py

# Grade-check (no transformers/datasets needed):
python grade.py
```

All generators are seeded deterministically per item_id (see
`scripts/common.py:rng_for`), so re-running a build script reproduces byte-
identical items.
