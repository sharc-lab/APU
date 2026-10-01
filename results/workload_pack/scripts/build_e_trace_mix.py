"""Family (e): context-length mix -- INTENDED to match the real trace CDF.

Task spec says to sample from results/traces/agent_step_lengths.parquet
(described as "real, committed") matching the real nebius-rebench-openhands /
SWE-Gym empirical CDF. Before writing this file we searched the whole
repository (git ls-files, filesystem find) for any *.parquet file and for
"agent_step_lengths", "agent_traces", "nebius", "rebench", "openhands",
"swe-gym"/"swegym": none of these exist anywhere in this repo. There is no
results/traces/ directory at all, committed or otherwise, and no trace file
to sample from.

Because that input genuinely does not exist, this family CANNOT be built
from a real trace CDF, and we are not going to fabricate one and pass it off
as real. Instead we build a clearly-labeled SYNTHETIC SUBSTITUTE distribution
chosen to be plausible for agentic coding step lengths: a lognormal, right-
skewed distribution over prompt tokens, clipped to [500, 120000]. Every item
in this family is tagged "length_distribution_source": "synthetic_substitute"
and the pack's README repeats this caveat. The README.md and the final report
both state the length comparison honestly: this family's CDF is compared
against itself, not against a real trace, because no real trace file exists
in the repository to compare against.

Items reuse family (a)'s needle-in-haystack document generator (same
deterministic exact_substring grading) so only the length-sampling procedure
differs between (a) and (e).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_a_longdoc_qa import build_document  # noqa: E402
from common import ITEMS_DIR, count_tokens, rng_for, write_jsonl  # noqa: E402

FAMILY = "trace_length_mix"
N_ITEMS = 60

# Synthetic substitute lognormal parameters (median ~3000 tokens, heavy
# right tail reaching into the 100K+ range, roughly in line with commonly
# reported agentic-coding step-length shapes -- NOT fit to any real trace).
LOGNORM_MU = math.log(3000)
LOGNORM_SIGMA = 1.35
MIN_TOKENS = 500
MAX_TOKENS = 120000


def sample_length(rng) -> int:
    v = rng.lognormvariate(LOGNORM_MU, LOGNORM_SIGMA)
    return int(min(max(v, MIN_TOKENS), MAX_TOKENS))


def difficulty_for(tokens: int) -> str:
    if tokens < 4000:
        return "short"
    if tokens < 20000:
        return "medium"
    if tokens < 60000:
        return "long"
    return "very_long"


def build_items() -> list[dict]:
    items = []
    for i in range(N_ITEMS):
        item_id = f"trace_mix_{i:03d}"
        rng = rng_for(item_id)
        target = sample_length(rng)
        needle_code = f"{rng.randrange(10**8, 10**9)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
        prompt, n_tokens = build_document(rng, target, needle_code)
        items.append(
            {
                "item_id": item_id,
                "family": FAMILY,
                "task_type": "needle_qa",
                "difficulty": difficulty_for(n_tokens),
                "target_prompt_tokens": target,
                "prompt_tokens": n_tokens,
                "prompt": prompt,
                "oracle_answer": needle_code,
                "grading": {"method": "exact_substring", "case_sensitive": True},
                "length_distribution_source": "synthetic_substitute",
            }
        )
    return items


def main():
    items = build_items()
    write_jsonl(ITEMS_DIR / "e_trace_mix.jsonl", items)
    print(f"[e] wrote {len(items)} trace-mix items (synthetic substitute distribution)")


if __name__ == "__main__":
    main()
