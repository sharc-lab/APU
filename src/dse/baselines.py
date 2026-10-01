"""DSE baselines: all-local, all-cloud (cheap/strong), and RouteLLM.

All baselines run against src/cloud/client.py's CloudClient. With no CLOUD_API_KEY set (the default in this
environment, and the only mode this task is in scope for), CloudClient is in stub mode: no network call is ever
made, every result carries stub=True, and cost_usd is always 0.0. See src/cloud/client.py's module docstring for
the full stub contract.

RouteLLM
--------
Checked 2026-09-30: `routellm` is NOT in pyproject.toml's dependencies and is not installed in this environment
(`python -c "import routellm"` raises ModuleNotFoundError). It needs to be added to pyproject.toml's
[project].dependencies before routellm_bert_baseline can do anything beyond returning a clearly-labeled "skipped,
package not installed" result for every task.

RouteLLM ships several routers (https://github.com/lm-sys/RouteLLM): mf (matrix-factorization) and sw_ranking
both require OpenAI embeddings at routing time, which means the routellm package needs OPENAI_API_KEY set overall
even though the embedding calls are a separate billing line from the actual chat completion. This task's scope is
the stub cloud provider only, with no real API key of any kind -- so mf and sw_ranking are skipped outright, not
attempted-and-caught, because they cannot even be configured without a key. The `bert` router is the one RouteLLM
router that does NOT need an OpenAI key (it scores with a small locally-run classifier), so it is the router this
module actually uses. RouteLLM's design also supports a local Ollama weak model via LiteLLM
(`routellm.controller.Controller(routers=["bert"], strong_model="...", weak_model="ollama/...")`) -- this module
wires the weak/strong model ids to this project's own cheap/strong tier names (src/cloud/client.CLOUD_MODELS) so
the comparison is apples-to-apples with the all_cloud_cheap/all_cloud_strong baselines, but does not attempt a
real Ollama call (out of scope here, same as the rest of this task -- see live_validation_stub.py).

Threshold calibration: RouteLLM's bert router outputs a float "win rate" (strong model's predicted win probability
over the weak model) per request, and routes to the strong model above a threshold. calibrate_routellm_threshold
below picks that threshold so that, if every task in a representative sample were routed independently at that
threshold, the EXPECTED cloud spend (cheap-tier cost for below-threshold tasks, strong-tier cost for
above-threshold tasks) stays at or under src/cloud/client.DEFAULT_SPEND_CAP_USD over the sample. This is a
same-budget calibration, not a live-traffic calibration (no live traffic was run) -- stated explicitly in the
return value.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.cloud.client import CloudClient, CLOUD_MODELS, DEFAULT_SPEND_CAP_USD  # noqa: E402


@dataclass
class BaselineResult:
    baseline: str
    task_id: str
    target: str  # "local" | "cloud_cheap" | "cloud_strong" | "skipped"
    reason: str
    stub: bool
    cost_usd: float = 0.0
    content: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


def all_local(tasks: list[dict[str, Any]]) -> list[BaselineResult]:
    """Baseline 1: every task runs locally, unconditionally. No envelope check, no cloud call at all -- the
    naive "never leave the device" policy this project's own router (src/dse/router.py) is meant to beat on
    silent-failure avoidance and on quality under budget."""
    return [
        BaselineResult(baseline="all_local", task_id=t.get("task_id", f"task_{i}"), target="local",
                       reason="all_local baseline: every task is routed locally regardless of envelope "
                              "feasibility, quality regime, or budget", stub=True)
        for i, t in enumerate(tasks)
    ]


def _all_cloud(tasks: list[dict[str, Any]], client: CloudClient, tier: str) -> list[BaselineResult]:
    model_id = CLOUD_MODELS[tier]
    out = []
    for i, t in enumerate(tasks):
        prompt = t.get("prompt", "")
        result = client.call(model_id=model_id, messages=[{"role": "user", "content": prompt}],
                              expected_output_tokens=t.get("expected_output_tokens", 256))
        out.append(BaselineResult(
            baseline=f"all_cloud_{tier}", task_id=t.get("task_id", f"task_{i}"), target=f"cloud_{tier}",
            reason=f"all_cloud_{tier} baseline: every task is routed to the cloud {tier} tier "
                   f"({model_id}) regardless of envelope feasibility or difficulty",
            stub=result.stub, cost_usd=result.cost_usd, content=result.content, raw=result.raw,
        ))
    return out


def all_cloud_cheap(tasks: list[dict[str, Any]], client: CloudClient) -> list[BaselineResult]:
    """Baseline 2a: every task routed to the cheap cloud tier (gpt-4o-mini, src/cloud/client.CLOUD_MODELS)."""
    return _all_cloud(tasks, client, "cheap")


def all_cloud_strong(tasks: list[dict[str, Any]], client: CloudClient) -> list[BaselineResult]:
    """Baseline 2b: every task routed to the strong cloud tier (claude-sonnet-4-5)."""
    return _all_cloud(tasks, client, "strong")


def calibrate_routellm_threshold(sample_win_rates: list[float], client: CloudClient,
                                  avg_input_tokens: int = 500, avg_output_tokens: int = 256) -> dict[str, Any]:
    """Pick a bert-router win-rate threshold such that routing `sample_win_rates` at that threshold keeps the
    EXPECTED spend (cheap tier below threshold, strong tier at/above) within client.spend_cap_usd over the
    sample. Not a live calibration (see module docstring) -- a same-budget, same-sample-size proxy."""
    from src.cloud.client import estimate_cost_usd

    cheap_cost = estimate_cost_usd(CLOUD_MODELS["cheap"], avg_input_tokens, avg_output_tokens)
    strong_cost = estimate_cost_usd(CLOUD_MODELS["strong"], avg_input_tokens, avg_output_tokens)
    n = len(sample_win_rates)
    budget_per_task = client.spend_cap_usd / n if n else client.spend_cap_usd

    # Sweep candidate thresholds (every observed win rate, sorted descending) and pick the lowest threshold
    # (i.e. routing the most tasks to "strong") whose expected per-task spend still clears the per-task budget.
    candidates = sorted(set(sample_win_rates), reverse=True)
    best_threshold = 1.0  # default: never route to strong (safest under budget)
    for thr in candidates:
        frac_strong = sum(1 for w in sample_win_rates if w >= thr) / n if n else 0.0
        expected_cost = frac_strong * strong_cost + (1 - frac_strong) * cheap_cost
        if expected_cost <= budget_per_task:
            best_threshold = thr
        else:
            break
    return {
        "threshold": best_threshold,
        "budget_per_task_usd": budget_per_task,
        "spend_cap_usd": client.spend_cap_usd,
        "cheap_cost_usd_per_task": cheap_cost,
        "strong_cost_usd_per_task": strong_cost,
        "n_sample": n,
        "note": "same-budget calibration against client.spend_cap_usd over a sample of win rates, NOT a live "
                "traffic calibration -- no live RouteLLM traffic was run (routellm is not installed; see "
                "module docstring)",
    }


def routellm_bert_baseline(tasks: list[dict[str, Any]], client: CloudClient,
                            threshold: float = 0.5) -> list[BaselineResult]:
    """Baseline 3: RouteLLM's bert router (the only router usable without OPENAI_API_KEY; see module docstring).

    Tries to import routellm and build a real Controller(routers=["bert"], ...). If routellm is not installed
    (the real state of this environment as of 2026-09-30), every task gets a BaselineResult with
    target="skipped" and a reason explaining exactly why, rather than silently falling back to a fake router.
    """
    try:
        from routellm.controller import Controller  # noqa: PLC0415
    except ImportError:
        return [
            BaselineResult(
                baseline="routellm_bert", task_id=t.get("task_id", f"task_{i}"), target="skipped", stub=True,
                reason="routellm is not installed (not in pyproject.toml's dependencies; "
                       "python -c \"import routellm\" raises ModuleNotFoundError as of 2026-09-30). "
                       "Add 'routellm' to pyproject.toml's [project].dependencies to run this baseline for real. "
                       "mf and sw_ranking routers are skipped unconditionally in this module because they need "
                       "OpenAI embeddings (OPENAI_API_KEY), which is out of scope for this task's stub-only run.",
            )
            for i, t in enumerate(tasks)
        ]

    # Real path -- only reached if routellm is actually installed in the environment running this.
    controller = Controller(
        routers=["bert"],
        strong_model=CLOUD_MODELS["strong"],
        weak_model=f"ollama/{CLOUD_MODELS['cheap']}",  # local-weak-model-via-LiteLLM shape RouteLLM expects
    )
    out = []
    for i, t in enumerate(tasks):
        prompt = t.get("prompt", "")
        # RouteLLM's public `route` call returns the chosen model id directly (strong_model or weak_model) at
        # the given threshold -- this is the real interface shape per RouteLLM's own docs/controller.py.
        decision_model = controller.route(prompt=prompt, router="bert", threshold=threshold)
        target = "cloud_strong" if decision_model == CLOUD_MODELS["strong"] else "cloud_cheap"
        tier = "strong" if target == "cloud_strong" else "cheap"
        result = client.call(model_id=CLOUD_MODELS[tier], messages=[{"role": "user", "content": prompt}],
                              expected_output_tokens=t.get("expected_output_tokens", 256))
        out.append(BaselineResult(
            baseline="routellm_bert", task_id=t.get("task_id", f"task_{i}"), target=target,
            reason=f"routellm bert router at threshold={threshold}: routed to {decision_model}",
            stub=result.stub, cost_usd=result.cost_usd, content=result.content, raw=result.raw,
        ))
    return out


if __name__ == "__main__":
    import tempfile

    from src.cloud.client import print_stub_banner_if_needed

    demo_tasks = [
        {"task_id": "t1", "prompt": "Summarize this paragraph in one sentence."},
        {"task_id": "t2", "prompt": "Write a detailed design doc for a distributed rate limiter, covering "
                                     "consistency, failure modes, and a rollout plan."},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        client = CloudClient(api_key=None, ledger_path=Path(tmp) / "ledger.jsonl")
        print_stub_banner_if_needed(client)
        for name, fn in (("all_local", lambda: all_local(demo_tasks)),
                          ("all_cloud_cheap", lambda: all_cloud_cheap(demo_tasks, client)),
                          ("all_cloud_strong", lambda: all_cloud_strong(demo_tasks, client)),
                          ("routellm_bert", lambda: routellm_bert_baseline(demo_tasks, client))):
            print(f"\n== {name} ==")
            for r in fn():
                print(f"  {r.task_id}: target={r.target} stub={r.stub} cost_usd={r.cost_usd:.4f} reason={r.reason}")
        calib = calibrate_routellm_threshold([0.2, 0.5, 0.8, 0.95], client)
        print(f"\ncalibrate_routellm_threshold demo: {calib}")
