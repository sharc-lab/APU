"""B6(d)/(e): live validation and failure-scenario demonstrations -- STUBBED, explicitly, by design.

A REAL version of either of these needs:
  (d) live validation: real cloud API calls (a real CLOUD_API_KEY, real spend against DEFAULT_SPEND_CAP_USD) to
      check the router's/baselines' predicted quality and cost against what an actual cloud completion returns,
      AND/OR a real local model call (loaded llama-server/Ollama process on evo-t2s or evo-x2) to check the
      envelope model's predicted TTFT/decode/feasibility against a real run.
  (e) failure-scenario demonstrations: the same -- actually triggering a HARD_FAIL (real vkAllocateMemory
      failure), a real SILENT_SPILL, or a real truncation_cliff requires a real local model process running at
      a real over-budget context, which needs machine time on evo-t2s or evo-x2.

Both are out of scope for this task (no real cloud API key is used per the task's own instructions -- the stub
provider in src/cloud/client.py is used instead -- and no machine-time run was performed). This script does NOT
skip the deliverable silently: it runs every baseline and the router against the STUB provider only, prints
"BLOCKED: cloud key" first (the mandated banner), and prints exactly what a real run of each scenario would need.

Run: py -3.12 src/dse/live_validation_stub.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.cloud.client import CloudClient, print_stub_banner_if_needed  # noqa: E402
from src.dse.baselines import all_cloud_cheap, all_cloud_strong, all_local, routellm_bert_baseline  # noqa: E402
from src.dse.router import EnvelopeRouter  # noqa: E402

FAILURE_SCENARIOS = [
    {
        "name": "evo-t2s HARD_FAIL (A-24 memory budget)",
        "task": {"task_id": "scenario_hard_fail", "model_id": "qwen3-32b", "context_length": 120000,
                  "prompt": "placeholder prompt"},
        "machine": "evo-t2s", "runtime_policy": "llama_ngl99",
        "what_a_real_run_needs": (
            "a real llama-server process on evo-t2s, Vulkan backend, -ngl 99, loaded with qwen3-32b, actually "
            "issued a request at n_ctx=120000 and observing the real vkAllocateMemory ErrorOutOfDeviceMemory "
            "(claim A-24) -- this script only checks that the envelope model PREDICTS that outcome and that "
            "the router avoids it, not that the real failure reproduces live."
        ),
    },
    {
        "name": "evo-x2 truncation_cliff (R1b quality cliff) below floor",
        "task": {"task_id": "scenario_truncation", "model_id": "qwen3-8b", "context_length": 120000,
                  "prompt_tokens": 48000, "full_prompt_tokens": 120000, "prompt": "placeholder prompt"},
        "machine": "evo-x2", "runtime_policy": "ollama_default",
        "what_a_real_run_needs": (
            "a real Ollama server on evo-x2 loaded with qwen3:8b, sent a real session-growth prompt sequence "
            "(harness/t2s_r2_session_growth.py's own R2.3(d) classification path), and a real scorer run over "
            "the resulting (possibly truncated) output to confirm the real score matches the predicted 0.0 at "
            "ratio 0.4 -- this script only checks the router's predicted routing decision, not a real score."
        ),
    },
    {
        "name": "evo-x2 bandwidth-hog decode penalty (PX2 co-runner)",
        "task": {"task_id": "scenario_corunner_decode", "model_id": "qwen3-8b", "context_length": 2000,
                  "prompt": "placeholder prompt"},
        "machine": "evo-x2", "runtime_policy": "ollama_default", "co_runner_present": True,
        "co_runner_kind": "bandwidth",
        "what_a_real_run_needs": (
            "a real bw_hog.py process actually saturating memory bandwidth on evo-x2 while a real llama-server "
            "call runs, to confirm the real decode_tok_s drop matches the predicted ~0.92x -- this script only "
            "evaluates the already-fit H2_X2_DECODE_MULTIPLIER prediction, not a fresh live measurement."
        ),
    },
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        client = CloudClient(api_key=None, ledger_path=Path(tmp) / "ledger.jsonl")
        print_stub_banner_if_needed(client)

        print("\n=== (d) Baselines against the stub provider (NOT live validation -- see module docstring) ===")
        demo_tasks = [{"task_id": "demo1", "prompt": "Summarize this paragraph in one sentence."}]
        for name, fn in (("all_local", lambda: all_local(demo_tasks)),
                          ("all_cloud_cheap", lambda: all_cloud_cheap(demo_tasks, client)),
                          ("all_cloud_strong", lambda: all_cloud_strong(demo_tasks, client)),
                          ("routellm_bert", lambda: routellm_bert_baseline(demo_tasks, client))):
            for r in fn():
                print(f"  [{name}] {r.task_id}: target={r.target} stub={r.stub} reason={r.reason[:120]}...")

        print("\n=== (e) Failure-scenario demonstrations against the stub provider/envelope predictions only ===")
        for scenario in FAILURE_SCENARIOS:
            router = EnvelopeRouter(
                client, machine=scenario["machine"], runtime_policy=scenario["runtime_policy"],
                co_runner_present=scenario.get("co_runner_present", False),
                co_runner_kind=scenario.get("co_runner_kind", "bandwidth"),
            )
            decision = router.route(scenario["task"])
            print(f"\n  scenario: {scenario['name']}")
            print(f"    router decision: target={decision.target} blocked_silent_failure="
                  f"{decision.blocked_silent_failure}")
            print(f"    reason: {decision.reason}")
            print(f"    REAL run needs: {scenario['what_a_real_run_needs']}")

        print("\n=== Summary ===")
        print("Everything above ran against the stub cloud provider and the envelope model's PREDICTIONS only. "
              "No real cloud API call and no real local model process was used anywhere in this script, per "
              "this task's scope (no real cloud API key; no machine time). A real (d)/(e) run needs both.")


if __name__ == "__main__":
    main()
