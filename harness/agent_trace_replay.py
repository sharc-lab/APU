"""Agent-trace replay workload (docs/EVALSET_DESIGN.md, "Test suite expansion", part (c), 2026-10-13).

Per-step prompt lengths come from results/traces/agent_step_lengths.parquet, uncensored sources only (the two
datasets in analysis/trace_weighted_pack.py UNCENSORED_DATASETS; the censored two are excluded for the reason given
in that module and in analysis/agent_traces.py). scripts/deploy_evo.py deploys files flat and pandas is not assumed on
the machines, so the per-step lengths live in harness/agent_trace_steps_uncensored.json, exported once by
`py -3.12 harness/agent_trace_replay.py --export` (tests/test_agent_trace_replay.py rebuilds it from the parquet and
requires equality).

Pure functions:
  load_schedule(path)                 -> {"trajectories": [{"id", "dataset", "tokens_qwen": [...], "tokens_llama": [...]}]}
  eligible(schedule, min_steps)       -> trajectories with at least min_steps steps, sorted by id
  sample_trajectory(schedule, seed)   -> one eligible trajectory, chosen with random.Random(seed)
  targets_for(traj, model, max_turns) -> the per-turn target prompt lengths (tokens_llama for llama models)
  filler_tokens_for_turn(...)         -> the closed-loop filler size for one turn
  positive_seed(schedule, ...)        -> the first seed whose trajectory exceeds the positive-control window early
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEDULE_FILE = HERE / "agent_trace_steps_uncensored.json"
# Same values as analysis/trace_weighted_pack.UNCENSORED_DATASETS (tests check they stay equal).
UNCENSORED_DATASETS = (
    "nebius/SWE-rebench-openhands-trajectories",
    "SWE-Gym/OpenHands-Sampled-Trajectories",
)
MIN_STEPS = 15          # three canary checks (turns 5, 10, 15)
MAX_TURNS = 40          # x2_r2_agent.N_CANARIES checks of every 5 turns
POSITIVE_WINDOW = 8192
POSITIVE_MARGIN = 1.25  # the positive-control trajectory exceeds 1.25 x the window by POSITIVE_BY_TURN
POSITIVE_BY_TURN = 9    # so the canary checks at turns 10 and 15 are past the window
SEED_SEARCH_START = 20261001


def build_schedule_from_parquet(parquet_path) -> dict:
    """The JSON payload from the parquet (needs pandas). Trajectories sorted by id, steps in step_idx order."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    u = df[df["dataset"].isin(UNCENSORED_DATASETS)].sort_values(["trajectory_id", "step_idx"])
    out = []
    for tid, g in u.groupby("trajectory_id", sort=True):
        steps = [int(s) for s in g["step_idx"]]
        if steps != list(range(1, len(steps) + 1)):
            raise ValueError(f"trajectory {tid}: step_idx not 1..n ({steps[:5]}...)")
        out.append({"id": str(tid), "dataset": str(g["dataset"].iloc[0]),
                    "tokens_qwen": [int(x) for x in g["tokens_qwen"]],
                    "tokens_llama": [int(x) for x in g["tokens_llama"]]})
    return {"source": "results/traces/agent_step_lengths.parquet", "datasets": list(UNCENSORED_DATASETS),
            "n_trajectories": len(out), "trajectories": out}


def load_schedule(path=SCHEDULE_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def eligible(schedule: dict, min_steps: int = MIN_STEPS) -> list[dict]:
    return sorted((t for t in schedule["trajectories"] if len(t["tokens_qwen"]) >= min_steps), key=lambda t: t["id"])


def sample_trajectory(schedule: dict, seed: int, min_steps: int = MIN_STEPS) -> dict:
    """One eligible trajectory, a pure function of (schedule, seed)."""
    pool = eligible(schedule, min_steps)
    if not pool:
        raise ValueError(f"no trajectory with >= {min_steps} steps")
    return random.Random(seed).choice(pool)


def token_column(model: str) -> str:
    """tokens_llama for a llama model tag, tokens_qwen otherwise (the parquet's two real tokenizers)."""
    return "tokens_llama" if "llama" in (model or "").lower() else "tokens_qwen"


def targets_for(traj: dict, model: str, max_turns: int = MAX_TURNS) -> list[int]:
    """Target prompt length (model tokens) for turns 1..min(steps, max_turns)."""
    return list(traj[token_column(model)][:max_turns])


def filler_tokens_for_turn(target_tokens: int, current_est: int, turn_text_est: int, calib_ratio: float | None,
                           overhead_est: int = 8) -> int:
    """Closed-loop filler size (chars/4 estimate units) so that the calibrated prompt of this turn's first call is
    about target_tokens: target / ratio, minus the transcript already there, the turn text and per-message overhead.
    Never negative (a target below the current transcript leaves an overshoot, recorded by the caller)."""
    ratio = calib_ratio if calib_ratio else 1.0
    return max(0, int(round(target_tokens / ratio)) - current_est - turn_text_est - overhead_est)


def positive_seed(schedule: dict, model: str = "qwen3:14b", window: int = POSITIVE_WINDOW,
                  margin: float = POSITIVE_MARGIN, by_turn: int = POSITIVE_BY_TURN, start: int = SEED_SEARCH_START,
                  tries: int = 1000) -> int:
    """The first seed >= start whose sampled trajectory's target exceeds margin x window by turn by_turn, for both
    tokenizer columns (so one seed serves every model)."""
    for s in range(start, start + tries):
        t = sample_trajectory(schedule, s)
        if all(targets_for(t, m)[by_turn - 1] > margin * window for m in ("llama", "qwen")):
            return s
    raise ValueError("no positive-control seed found")


def fidelity(rows: list[dict], tol: float = 0.10) -> dict:
    """Trace-fidelity on replay turn rows: share of reachable turns (target above the turn's floor, i.e. the transcript
    before filler was below the target) whose calibrated first-call prompt is within tol of the target."""
    reach = [r for r in rows if r.get("replay_target_tokens") and r.get("replay_reachable")]
    ok = [r for r in reach if abs(r["replay_prompt_calibrated"] - r["replay_target_tokens"])
          <= tol * r["replay_target_tokens"]]
    return {"n_reachable": len(reach), "n_within": len(ok), "rate": (len(ok) / len(reach)) if reach else None,
            "tol": tol}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["--export"]:
        print(__doc__)
        return 2
    repo = HERE.parent
    payload = build_schedule_from_parquet(repo / "results" / "traces" / "agent_step_lengths.parquet")
    SCHEDULE_FILE.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {SCHEDULE_FILE.name}: {payload['n_trajectories']} trajectories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
