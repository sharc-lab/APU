"""Agent workload trace CDF: per-step context length from two public agent trajectory datasets, tokenized with
real qwen3 and llama3.1 tokenizers.

Datasets (streaming=True, no full download):
  (a) nebius/SWE-agent-trajectories -- column "trajectory", a list of {role, text|system_prompt, mask,
      cutoff_date} dicts per trajectory. role in {system, user, ai}. The system row's content is under
      "system_prompt" instead of "text".
  (b) Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k -- column "messages", a standard
      [{role, content}, ...] list per trajectory.

Per trajectory, per step i (i = index of a user/assistant/ai boundary, one step per assistant/ai turn -- a "step"
is what the runtime would actually send as a prompt right before that turn's response): reconstruct the full
message history up to and including that step (system prompt first, then every message before this turn),
tokenize the concatenated text with both tokenizers, and record system+tool-schema token count (the system
message alone) separately from the full cumulative count.

No model weights are loaded, tokenizers only (AutoTokenizer.from_pretrained, PyTorch absent is fine and expected).

Usage: py -3.12 analysis/agent_traces.py
Outputs: results/traces/agent_step_lengths.parquet, figures/fig_trace_cdf.png
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_TRAJECTORIES = 600
TIMEBOX_S = 300  # 5 minutes per dataset per the instruction: stop and move on if streaming stalls

QWEN_MODEL = "Qwen/Qwen3-8B"
LLAMA_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

THRESHOLDS = (8_000, 16_000, 32_000, 40_960, 64_000, 128_000)


def load_tokenizers():
    from transformers import AutoTokenizer
    tok_qwen = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tok_llama = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    return tok_qwen, tok_llama


def count_tokens(tok, text):
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)["input_ids"])


def iter_nebius_steps(row):
    """Yields (step_idx, system_text, cumulative_text) for each assistant ("ai") turn in a nebius trajectory."""
    traj = row.get("trajectory") or []
    system_text = ""
    history_parts = []
    step_idx = 0
    for m in traj:
        role = m.get("role")
        text = m.get("system_prompt") if role == "system" else m.get("text")
        text = text or ""
        if role == "system":
            system_text = text
            history_parts.append(text)
            continue
        history_parts.append(text)
        if role == "ai":
            step_idx += 1
            cumulative = "\n".join(history_parts)
            yield step_idx, system_text, cumulative


def iter_swesmith_steps(row):
    """Yields (step_idx, system_text, cumulative_text) for each assistant turn in a SWE-smith trajectory."""
    msgs = row.get("messages") or []
    system_text = ""
    history_parts = []
    step_idx = 0
    for m in msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system_text = content
            history_parts.append(content)
            continue
        history_parts.append(content)
        if role == "assistant":
            step_idx += 1
            cumulative = "\n".join(history_parts)
            yield step_idx, system_text, cumulative


DATASETS = {
    "nebius/SWE-agent-trajectories": iter_nebius_steps,
    "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k": iter_swesmith_steps,
}


def process_dataset(name, step_iter_fn, tok_qwen, tok_llama, n_trajectories=N_TRAJECTORIES, timebox_s=TIMEBOX_S):
    from datasets import load_dataset

    rows = []
    t0 = time.monotonic()
    try:
        ds = load_dataset(name, split="train", streaming=True)
        print(f"{name}: starting stream", flush=True)
        it = iter(ds)
        n_traj_done = 0
        for traj_idx in range(n_trajectories):
            if time.monotonic() - t0 > timebox_s:
                print(f"{name}: timeboxed at {timebox_s}s, stopping with {n_traj_done} trajectories done", flush=True)
                break
            fetch_t0 = time.monotonic()
            try:
                row = next(it)
            except StopIteration:
                print(f"{name}: dataset exhausted at {n_traj_done} trajectories", flush=True)
                break
            fetch_dt = time.monotonic() - fetch_t0
            n_steps_this_traj = 0
            steps_buffer = []
            for step_idx, system_text, cumulative in step_iter_fn(row):
                tq = count_tokens(tok_qwen, cumulative)
                tl = count_tokens(tok_llama, cumulative)
                sq = count_tokens(tok_qwen, system_text)
                sl = count_tokens(tok_llama, system_text)
                steps_buffer.append({
                    "dataset": name, "trajectory_id": row.get("instance_id", f"traj{traj_idx}"),
                    "step_idx": step_idx, "tokens_qwen": tq, "tokens_llama": tl,
                    "system_tokens_qwen": sq, "system_tokens_llama": sl,
                })
                n_steps_this_traj += 1
            for r in steps_buffer:
                r["n_steps_total"] = n_steps_this_traj
            rows.extend(steps_buffer)
            n_traj_done += 1
            if n_traj_done % 25 == 0 or fetch_dt > 5.0:
                print(f"{name}: {n_traj_done}/{n_trajectories} trajectories, {len(rows)} steps so far, "
                     f"{time.monotonic() - t0:.1f}s elapsed, last fetch {fetch_dt:.1f}s", flush=True)
    except Exception as e:
        print(f"{name}: ERROR after {time.monotonic() - t0:.1f}s, {len(rows)} rows collected so far:")
        traceback.print_exc()
        return rows, repr(e)
    print(f"{name}: {n_traj_done} trajectories, {len(rows)} steps, {time.monotonic() - t0:.1f}s")
    return rows, None


def build_parquet(all_rows, out_path):
    import pandas as pd
    df = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def build_cdf_plot(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in df.groupby("dataset"):
        vals = np.sort(group["tokens_qwen"].to_numpy())
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=f"{name} (qwen3 tokenizer)")
    ax.set_xscale("log")
    ax.set_xlabel("step context length (tokens, qwen3 tokenizer)")
    ax.set_ylabel("CDF")
    ax.set_title("Agent-step context length, real trajectories")
    ax.legend(fontsize=8)
    for t in THRESHOLDS:
        ax.axvline(t, color="gray", linestyle=":", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(df):
    import numpy as np
    out = {}
    for name, group in df.groupby("dataset"):
        n_traj = group["trajectory_id"].nunique()
        n_steps = len(group)
        steps_per_traj = group.groupby("trajectory_id").size()
        vals_q = group["tokens_qwen"].to_numpy()
        vals_l = group["tokens_llama"].to_numpy()
        frac_over = {t: float((vals_q > t).mean()) for t in THRESHOLDS}
        traj_max = group.groupby("trajectory_id")["tokens_qwen"].max()
        frac_traj_over = {t: float((traj_max > t).mean()) for t in (32_000, 40_960, 128_000)}
        out[name] = {
            "n_trajectories": int(n_traj), "n_steps": int(n_steps),
            "steps_per_traj_p50": float(np.percentile(steps_per_traj, 50)),
            "steps_per_traj_p90": float(np.percentile(steps_per_traj, 90)),
            "qwen_p50": float(np.percentile(vals_q, 50)), "qwen_p90": float(np.percentile(vals_q, 90)),
            "qwen_p99": float(np.percentile(vals_q, 99)),
            "llama_p50": float(np.percentile(vals_l, 50)), "llama_p90": float(np.percentile(vals_l, 90)),
            "llama_p99": float(np.percentile(vals_l, 99)),
            "frac_steps_over_threshold": frac_over,
            "frac_trajectories_ever_over": frac_traj_over,
        }
    return out


def main():
    tok_qwen, tok_llama = load_tokenizers()
    all_rows = []
    errors = {}
    for name, fn in DATASETS.items():
        rows, err = process_dataset(name, fn, tok_qwen, tok_llama)
        all_rows.extend(rows)
        if err:
            errors[name] = err

    if not all_rows:
        print("no rows collected from any dataset, nothing to write")
        return

    df = build_parquet(all_rows, ROOT / "results" / "traces" / "agent_step_lengths.parquet")
    build_cdf_plot(df, ROOT / "figures" / "fig_trace_cdf.png")
    summary = summarize(df)

    print("\n=== SUMMARY ===")
    for name, s in summary.items():
        print(f"\n{name}:")
        print(f"  trajectories={s['n_trajectories']} steps={s['n_steps']}")
        print(f"  steps/traj p50={s['steps_per_traj_p50']:.0f} p90={s['steps_per_traj_p90']:.0f}")
        print(f"  qwen3 tokens p50={s['qwen_p50']:.0f} p90={s['qwen_p90']:.0f} p99={s['qwen_p99']:.0f}")
        print(f"  llama3.1 tokens p50={s['llama_p50']:.0f} p90={s['llama_p90']:.0f} p99={s['llama_p99']:.0f}")
        print("  fraction of steps over threshold (qwen3 tokenizer):")
        for t, f in s["frac_steps_over_threshold"].items():
            print(f"    {t}: {f:.3f}")
        print("  fraction of trajectories that EVER exceed (qwen3 tokenizer):")
        for t, f in s["frac_trajectories_ever_over"].items():
            print(f"    {t}: {f:.3f}")
    if errors:
        print("\n=== ERRORS ===")
        for name, e in errors.items():
            print(f"{name}: {e}")


if __name__ == "__main__":
    main()
