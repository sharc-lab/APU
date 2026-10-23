"""harness/demo_overflow.py: the cut-off detector reads the server's total prompt (cached prefix included), so prefix
caching on a long session does not make it fire; message drops, token cuts and context shifts do. Checked on the
committed R2 mechanism run (Ollama 0.34.4, OLLAMA_DEBUG=1): its per-call rows (results/x2_r2_mechanism.jsonl, the
mechanism job's own verdict per call is the ground truth) and its default-window server log (40 turns, nothing cut)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import demo_overflow as D  # noqa: E402

LOGS = REPO / "results" / "x2_r2_mechanism_logs"
DEFAULT_LOG = LOGS / "llama3.1_8b_ollama_default_call2_notools_20260901.ollama_log.txt"
EVAL_RE = re.compile(r"prompt eval time = .*?/\s*(\d+) tokens")


@pytest.fixture(scope="module")
def calls():
    with open(REPO / "results" / "x2_r2_mechanism.jsonl", encoding="utf-8") as fh:
        return [r for r in map(json.loads, fh) if r.get("record") == "r2m_call"]


def _verdict(r, own_scale=1.05):
    n_tok = r["new_prompt"][-1]["n_tokens"]
    return D.detect(round(n_tok * own_scale), r["prompt_eval_count"], "\n".join(r["log_excerpt"]),
                    r["num_ctx_requested"] or 131072, n_messages=r["n_messages_sent"])


def truly_cut(r) -> bool:
    return bool(r["message_truncation_in_log"] or r["context_shifts"] or r["token_cut"] or r["exceed_context_error"])


def test_agrees_with_the_mechanism_runs_own_verdict_on_every_call(calls):
    assert len(calls) == 400
    for r in calls:
        assert _verdict(r)["cut"] == truly_cut(r), (r["arm_id"], r["turn_idx"], r["call_idx"])
    assert sum(truly_cut(r) for r in calls if r["arm_id"].startswith("ollama_ctx_4096")) == 76
    assert not any(truly_cut(r) for r in calls if r["arm_id"].startswith("ollama_default"))


def test_default_window_long_session_does_not_fire_despite_prefix_caching():
    slices, cur = [], []
    for line in DEFAULT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        cur.append(line)
        if "stop processing:" in line:
            slices.append("\n".join(cur))
            cur = []
    slices = [s for s in slices if "new prompt" in s]
    assert len(slices) >= 80
    cached = naive = 0
    for s in slices:
        u = D.used_from_log(s)
        evaluated = int(EVAL_RE.findall(s)[-1])     # what a cache-blind prompt_eval_count would report
        own = round(u["n_tokens"] * 1.05)            # an own-count estimate 5% high
        assert not D.detect(own, u["n_tokens"], s, 131072)["cut"]
        cached += (u["cached"] or 0) > 0
        naive += D.naive_detect(own, evaluated)
    assert cached >= len(slices) // 2 and naive >= len(slices) // 2


def test_count_shortfall_alone_fires():
    log = ("slot   operator(): id  0 | task 9 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 3900\n"
           "slot   operator(): id  0 | task 9 | cached n_tokens = 3000, memory_seq_rm [3000, end)\n")
    v = D.detect(9000, 3900, log, 4096)
    assert v["cut"] and v["used_tokens"] == 3900 and v["cached_tokens"] == 3000
    assert v["prompt_eval_count_is_total"] is True
    assert not D.detect(4000, 3900, log, 4096)["cut"]      # within the 10% tolerance


def test_truncation_line_counts_as_a_drop_only_when_messages_were_dropped():
    line = ('time=x level=DEBUG source=prompt.go:77 msg="truncating input messages which exceed context length" '
            'truncated=8\n')
    assert not D.detect(100, 100, line, 4096, n_messages=8)["cut"]
    assert D.detect(100, 100, line, 4096, n_messages=12)["cut"]


def test_no_log_falls_back_to_the_window_never_to_prompt_eval_count():
    v = D.detect(9000, 100, None, 40960)
    assert v["method"] == "window_only" and not v["cut"]
    assert D.detect(9000, 9000, None, 4096)["cut"]
