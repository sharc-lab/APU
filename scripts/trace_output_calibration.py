"""Measured output tokens per agent step and an Anthropic-tokenizer calibration for the two uncensored trace sets
(row cloud-cost-per-real-task, operator follow-up 2026-10-21).

Phase --extract (no key, public Hugging Face streams): re-streams the same rows analysis/agent_traces.py sampled
(first rows of each stream, no shuffle; nebius/SWE-rebench-openhands-trajectories split train and
SWE-Gym/OpenHands-Sampled-Trajectories split train.raw), rebuilds the same trajectory ids, and for every assistant
turn counts the qwen3 tokens of what the model generated at that step (content + tool_calls JSON, the same "piece"
agent_traces.iter_openhands_steps appends to the history). Checks each trajectory's step count and final-step
tokens_qwen against results/traces/agent_step_lengths.parquet and refuses to write on any mismatch. Writes
results/traces/agent_step_outputs.parquet and, for a fixed-seed sample of SAMPLE_PER_SET trajectories per set, the
final-step full prompt text to a local scratch directory (not committed: raw dataset text).

Phase --count (needs ANTHROPIC_API_KEY in this process's environment; never printed or written): one
messages.count_tokens call per sampled trajectory (at most MAX_CALLS in total), the final-step prompt sent as one user
message. Appends one record per call to results/anthropic_token_calibration_20261021.jsonl. count_tokens is not
billed. The reported count includes a few tokens of message framing; against prompts of thousands of tokens it is
not removed.

Usage: py -3.12 scripts/trace_output_calibration.py --extract --scratch DIR
       py -3.12 scripts/trace_output_calibration.py --count --scratch DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

STEPS_FILE = REPO / "results" / "traces" / "agent_step_lengths.parquet"
OUT_FILE = REPO / "results" / "traces" / "agent_step_outputs.parquet"
CALIB_FILE = REPO / "results" / "anthropic_token_calibration_20261021.jsonl"
SETS = ("nebius/SWE-rebench-openhands-trajectories", "SWE-Gym/OpenHands-Sampled-Trajectories")
SAMPLE_PER_SET = 20
SAMPLE_SEED = 20261021
MAX_CALLS = 40
COUNT_MODEL = "claude-sonnet-5-5"


def step_outputs(row):
    """[(step_idx, assistant piece text)] in the same order agent_traces.iter_openhands_steps yields steps."""
    import agent_traces as at
    msgs = row.get("trajectory") if row.get("trajectory") is not None else row.get("messages")
    out, k = [], 0
    for m in msgs or []:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        tool_text = at._tool_calls_text(m.get("tool_calls"))
        k += 1
        out.append((k, (content + "\n" + tool_text) if tool_text else content))
    return out


def sample_ids(ids, n=SAMPLE_PER_SET, seed=SAMPLE_SEED):
    ids = sorted(ids)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def extract(scratch: Path):
    import pandas as pd
    from datasets import load_dataset
    import agent_traces as at
    tok_qwen, _ = at.load_tokenizers()
    ref = pd.read_parquet(STEPS_FILE)
    rows, mismatches = [], []
    scratch.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        spec = at.DATASETS[name]
        sub = ref[ref["dataset"] == name]
        want = set(sub["trajectory_id"])
        final = sub.sort_values("step_idx").groupby("trajectory_id").last()
        picked = set(sample_ids(want))
        ds = load_dataset(name, split=spec["split"], streaming=True)
        it, seen = iter(ds), 0
        for i in range(10 ** 6):
            if seen == len(want):
                break
            row = next(it)
            tid = spec["traj_id_fn"](row, i)
            if tid not in want:
                continue
            seen += 1
            steps = list(spec["iter_fn"](row))
            outs = step_outputs(row)
            last_cum = steps[-1][2] if steps else ""
            tq_final = at.count_tokens(tok_qwen, last_cum)
            f = final.loc[tid]
            if len(steps) != int(f["n_steps_total"]) or tq_final != int(f["tokens_qwen"]) or len(outs) != len(steps):
                mismatches.append((tid, len(steps), int(f["n_steps_total"]), tq_final, int(f["tokens_qwen"])))
                continue
            for k, text in outs:
                rows.append({"dataset": name, "trajectory_id": tid, "step_idx": k,
                             "output_tokens_qwen": at.count_tokens(tok_qwen, text)})
            if tid in picked:
                h = hashlib.sha256(tid.encode()).hexdigest()[:16]
                (scratch / f"{h}.json").write_text(json.dumps({"dataset": name, "trajectory_id": tid,
                                                               "qwen_tokens": tq_final, "text": last_cum}),
                                                   encoding="utf-8")
            if seen % 25 == 0:
                print(f"{name}: {seen}/{len(want)}", flush=True)
        print(f"{name}: matched {seen}/{len(want)}", flush=True)
    if mismatches:
        print(f"REFUSED: {len(mismatches)} trajectories do not match the step-length file, e.g. {mismatches[:3]}")
        sys.exit(2)
    pd.DataFrame(rows).to_parquet(OUT_FILE, index=False)
    print(f"wrote {OUT_FILE.relative_to(REPO)} ({len(rows)} steps)")


def count(scratch: Path, client=None):
    files = sorted(scratch.glob("*.json"))
    if len(files) > MAX_CALLS:
        raise SystemExit(f"refused: {len(files)} samples > {MAX_CALLS} calls")
    done = set()
    if CALIB_FILE.exists():
        done = {json.loads(l)["trajectory_id"] for l in CALIB_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}
    if len(done) + len([f for f in files if json.loads(f.read_text(encoding="utf-8"))["trajectory_id"] not in done]) \
            > MAX_CALLS:
        raise SystemExit("refused: would exceed the call cap")
    if client is None:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from this process only
    n = 0
    with open(CALIB_FILE, "a", encoding="utf-8") as out:
        for f in files:
            s = json.loads(f.read_text(encoding="utf-8"))
            if s["trajectory_id"] in done:
                continue
            r = client.messages.count_tokens(model=COUNT_MODEL, messages=[{"role": "user", "content": s["text"]}])
            n += 1
            out.write(json.dumps({"record": "count_tokens", "ts_utc": datetime.now(timezone.utc).isoformat(),
                                  "endpoint": "messages.count_tokens", "model": COUNT_MODEL,
                                  "dataset": s["dataset"], "trajectory_id": s["trajectory_id"],
                                  "text_sha256": hashlib.sha256(s["text"].encode("utf-8")).hexdigest(),
                                  "chars": len(s["text"]), "qwen_tokens": s["qwen_tokens"],
                                  "anthropic_tokens": int(r.input_tokens),
                                  "basis": "final-step full prompt (agent_traces cumulative text) as one user message"})
                      + "\n")
            out.flush()
    print(f"{n} count_tokens calls recorded in {CALIB_FILE.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--scratch", required=True, type=Path)
    a = ap.parse_args()
    if a.extract:
        extract(a.scratch)
    if a.count:
        count(a.scratch)
