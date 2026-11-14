"""Sizing check for the token-matched long tasks (longtask-tokens-v2 padding, written 2026-11-13). Pure: no network,
no model call. Needs the Qwen3 tokenizer.json (Hugging Face, public; the 2507 model's file is byte-identical to
Qwen3-8B's, sha256 aeb13307a71acd8f...).

For every task of the given seed, the reference (oracle) transcript is padded two ways and its final prompt is counted
with the model's tokenizer as the Qwen3 chat template frames it (harness/longtask_tokens.TokenizerSizing):
  v1  pads sized by the harness estimate (chars/4 x PROJ_CALIB), as run in longtask-tokens-v1;
  v2  pads sized by the tokenizer (the smallest line count reaching each trace target).
Each is set against the matched trace's final-step size (trace_max, Qwen tokens, agent_trace_steps_uncensored.json).

v1 ran with seeds 20260907..09 and v2 is pre-registered with seeds 20260911..13; the padding text depends on the task
and step, not on the seed's env values beyond record ids, so the check is run on both seed sets.

Writes results/longtask_v2_sizing_check_v1.jsonl. Run: py -3.12 analysis/longtask_v2_sizing_check.py --tokenizer-json F
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
OUT = REPO / "results" / "longtask_v2_sizing_check_v1.jsonl"
SEEDS = (20260907, 20260911)
OH = [f"lt_openhands_{i:02d}" for i in range(1, 11)]


def v1_final_real(T, L, task, env, sz) -> int:
    pad = T.Padder(task, env)
    e = copy.deepcopy(env)
    msgs = [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": L.user_message(task)}]
    for k, c in enumerate(T._reference_calls(task, env)):
        res = pad(k, L.execute(e, c))
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": dict(c)}]})
        msgs.append(L.tool_message(c, res))
    return sz.model_tokens(msgs)


def main(argv=None) -> int:
    import longtask_env as L
    import longtask_tokens as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-json", required=True)
    a = ap.parse_args(argv)
    sha = hashlib.sha256(Path(a.tokenizer_json).read_bytes()).hexdigest()
    sz = T.TokenizerSizing(tokenizer_json=a.tokenizer_json)
    rows = []
    for seed in SEEDS:
        for task, env in L.build_tasks(seed, OH):
            p2 = T.oracle_profile(task, env, sz)
            v1 = v1_final_real(T, L, task, env, sz)
            rows.append({"record": "sizing_check", "seed": seed, "task_id": task.task_id,
                         "trace_max": p2["trace_max"], "v1_final_model_tokens": v1,
                         "v1_ratio": round(v1 / p2["trace_max"], 4), "v2_final_model_tokens": p2["oracle_final"],
                         "v2_ratio": round(p2["final_ratio"], 4), "v2_oracle_ok": p2["oracle_ok"],
                         "tokenizer_sha256": sha})
            print(rows[-1])
    OUT.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
