"""Prints src/dse/router.Router's decisions, with reasons, on 10 sample steps of the R2 real run (x2_r2_real_v1).

Each sample replays its whole session through a Router matching the session's arm (evo-x2, Ollama default or
num_ctx N, the session's model and its own token calibration ratio), so cloud-budget state at the sampled step is
what the earlier cloud decisions of that replay left. Cloud client: stub, on an empty temporary ledger (no real call,
no real spend, and the output does not depend on results/cloud_ledger.jsonl). The samples span the three tiers of
the run: Ollama default (evo-x2 loads 131072 for llama3.1:8b, 40960 for qwen3:14b), num_ctx 32768, and num_ctx
4096 at early, mid and late turns. The last sample uses a zero cloud budget to show the fallback when the quality
floor needs dropped history and the cloud is unaffordable.

The "real run" figures on each sample are read from the same rows: the call's prompt estimate times the session's
calibration ratio, the context Ollama had loaded, and whether that prompt was over it.

Usage: py -3.12 analysis/router_sample_decisions.py   (prints, and writes results/router_sample_decisions.txt)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.cloud.client import CloudClient  # noqa: E402
from src.dse import r2_replay  # noqa: E402

R2_FILE = REPO / "results" / "x2_r2_real_v1.jsonl"
OUT = REPO / "results" / "router_sample_decisions.txt"
SEED = 20260901
BUDGET_USD = 5.0
QUALITY_FLOOR = 0.9
LATENCY_TARGET_MS = None  # not binding here: the fit is fresh-prompt TTFT, Ollama reuses the session's cached prefix

# (label, model, tier prefix of the arm id, turn, call, budget_usd)
SAMPLES = [
    ("Ollama default, llama3.1:8b, mid", "llama3.1:8b", "ollama_default", 20, 2, BUDGET_USD),
    ("Ollama default, qwen3:14b, before the real overflow", "qwen3:14b", "ollama_default", 24, 2, BUDGET_USD),
    ("Ollama default, qwen3:14b, canary turn", "qwen3:14b", "ollama_default", 30, 2, BUDGET_USD),
    ("num_ctx 32768, llama3.1:8b, early", "llama3.1:8b", "ollama_ctx_32768", 10, 2, BUDGET_USD),
    ("num_ctx 32768, llama3.1:8b, one turn before the real overflow", "llama3.1:8b", "ollama_ctx_32768", 22, 2,
     BUDGET_USD),
    ("num_ctx 4096, llama3.1:8b, early (turn 1)", "llama3.1:8b", "ollama_ctx_4096", 1, 1, BUDGET_USD),
    ("num_ctx 4096, llama3.1:8b, first real overflow turn", "llama3.1:8b", "ollama_ctx_4096", 3, 1, BUDGET_USD),
    ("num_ctx 4096, qwen3:14b, mid", "qwen3:14b", "ollama_ctx_4096", 12, 2, BUDGET_USD),
    ("num_ctx 4096, qwen3:14b, mid canary turn", "qwen3:14b", "ollama_ctx_4096", 20, 2, BUDGET_USD),
    ("num_ctx 4096, llama3.1:8b, late canary turn, zero cloud budget", "llama3.1:8b", "ollama_ctx_4096", 40, 2, 0.0),
]


def _session(sessions, model, tier):
    for (m, arm, seed, _), srows in sessions.items():
        if m == model and seed == SEED and (arm == tier or arm.startswith(tier + "_call2")):
            return srows
    raise KeyError((model, tier, SEED))


def run() -> str:
    rows = r2_replay.read_rows(R2_FILE)
    sessions = r2_replay.r2_sessions(rows)
    lines = [
        "Router sample decisions on R2 real run steps (results/x2_r2_real_v1.jsonl, seed "
        f"{SEED}), written by analysis/router_sample_decisions.py",
        f"Router settings: hardware evo-x2, runtime = the session's arm, quality_floor {QUALITY_FLOOR}, "
        f"latency_target_ms {LATENCY_TARGET_MS}, cloud model gpt-4o-mini, stub cloud client on an empty ledger, "
        "context margin 5%, answer budget 384 (the harness's num_predict). Canary turns need turn 1 (history tags).",
        "",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for k, (label, model, tier, turn, call, budget) in enumerate(SAMPLES, start=1):
            srows = _session(sessions, model, tier)
            client = CloudClient(mode="stub", ledger_path=Path(tmp) / f"ledger_{k}.jsonl")
            router = r2_replay.router_for_session(srows, budget_usd=budget, quality_floor=QUALITY_FLOOR,
                                                  latency_target_ms=LATENCY_TARGET_MS, cloud_client=client)
            d = next(x for x in router.replay_calls(srows) if x.turn_idx == turn and x.call_idx == call)
            o = d.observed
            t = d.tokens
            lat = "n/a" if d.predicted_latency_ms is None else f"{d.predicted_latency_ms:,.0f} ms"
            lines += [
                f"[{k}] {label}: {model} {srows[0]['arm_id']}, turn {turn} call {call}",
                f"    real run: prompt {o['real_prompt_tokens_calibrated']:,} tok (calibrated), loaded context "
                f"{o['loaded_context']:,}, over loaded window: {'yes' if o['real_over_loaded_window'] else 'no'}",
                f"    decision: target={d.target} machine={d.machine} runtime={d.runtime} model={d.model} "
                f"num_ctx={d.num_ctx:,}",
                f"    tokens: system {t['system']:,}, kept history {t['kept_history']:,}, answer budget "
                f"{t['answer_budget']:,}, total {t['total']:,} (untrimmed {t['full_total']:,}); "
                f"kept turns {d.kept_turns}, dropped turns {d.dropped_turns}",
                f"    predicted latency {lat}; est cost ${d.est_cost_usd:.4f}; budget remaining "
                f"${d.budget_remaining_usd:.4f}; stub={d.stub}; quality floor met: {d.quality_floor_met}",
                f"    reason: {d.reason}",
                "",
            ]
    return "\n".join(lines)


def main():
    text = run()
    print(text)
    OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
