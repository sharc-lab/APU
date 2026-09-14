"""The frozen slice configuration must stay consistent with the artifacts it pins.

A task list that drifts from its pinned hash, or a design that quietly drops below the
pre-registered cell size, invalidates comparability without any visible failure. These tests make
that drift loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from seam.agent.tools import load_workload
from seam.budget import PricingTable

REPO_ROOT = Path(__file__).resolve().parents[1]
SLICE_CONFIG = REPO_ROOT / "configs" / "mslice.yaml"


@pytest.fixture
def cfg() -> dict[str, Any]:
    loaded = yaml.safe_load(SLICE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_task_list_matches_its_pinned_hash(cfg: dict[str, Any]) -> None:
    """The freeze is only real if it is asserted."""
    workload = load_workload(REPO_ROOT / cfg["workload"]["task_list"])
    assert workload.sha256 == cfg["workload"]["task_list_sha256"]


def test_design_meets_the_pre_registered_minimums(cfg: dict[str, Any]) -> None:
    design = cfg["design"]
    assert design["n_tasks_per_cell"] >= 20
    assert design["noise_floor_repeats"] >= 20
    assert design["aa_repeats"] >= 20
    assert design["null_threshold_cv_multiple"] == 2.0
    assert design["randomize_execution_order"] is True


def test_at_least_four_deadlines_spanning_generous_to_tight(cfg: dict[str, Any]) -> None:
    multipliers = cfg["policy"]["deadline_multipliers"]
    assert len(multipliers) >= 4
    assert multipliers == sorted(multipliers, reverse=True)
    assert max(multipliers) >= 2.0 * min(multipliers)


def test_budget_ceilings_match_the_authorization(cfg: dict[str, Any]) -> None:
    budget = cfg["budget"]
    assert budget["slice_ceiling_usd"] == 10.00
    assert budget["project_ceiling_usd"] == 50.00


def test_cloud_model_is_a_pinned_identifier_not_a_floating_alias(cfg: dict[str, Any]) -> None:
    """AM-020: a pinned identifier in the provider's convention; `-latest` is always forbidden."""
    model_id = cfg["models"]["cloud"]["model_id"]
    assert not model_id.endswith("-latest")
    pricing = PricingTable.load(REPO_ROOT / cfg["models"]["cloud"]["pricing_table"])
    assert model_id in pricing.tiers_by_model
    assert pricing.pin_convention_by_model[model_id] == "anthropic_dateless_generation_id"


def test_policy_is_predictive_and_deadline_is_advisory(cfg: dict[str, Any]) -> None:
    """AM-022 items 1 and 3. Preemptive semantics are deferred, not silently adopted."""
    assert cfg["policy"]["kind"] == "predictive_local_first"
    assert cfg["policy"]["deadline_enforced"] is False


def test_slice_uses_cpu_targets_only(cfg: dict[str, Any]) -> None:
    """No iGPU, no NPU in this slice."""
    assert set(cfg["targets"]) == {"cpu-p", "cpu-lpe"}


def test_inference_threads_are_explicit_not_a_cpu_count_heuristic(cfg: dict[str, Any]) -> None:
    """Platform A is 8C/8T with 4 P + 4 LP-E; a heuristic would oversubscribe both arms."""
    assert cfg["openvino"]["inference_num_threads"] == 4
    assert cfg["openvino"]["enable_cpu_pinning"] is True
    assert cfg["openvino"]["scheduling_core_type"] == {
        "cpu-p": "PCORE_ONLY",
        "cpu-lpe": "ECORE_ONLY",
    }
