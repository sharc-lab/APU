"""Verify evaluation/probes/task_map.yaml covers every sdk_direct.py task.

Checks:
  - All 14 task IDs present (exactly one entry each).
  - Each entry has exactly one probe_id and one category.
  - Every probe_id exists in evaluation/probes/prompts.jsonl.
  - Category values match sdk_direct.py TASKS.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

REPO = pathlib.Path(__file__).parent.parent
TASK_MAP_FILE = REPO / "evaluation" / "probes" / "task_map.yaml"
PROMPTS_FILE = REPO / "evaluation" / "probes" / "prompts.jsonl"
SDK_DIRECT = REPO / "harness" / "adapters" / "sdk_direct.py"


def _load_task_map() -> dict:
    return yaml.safe_load(TASK_MAP_FILE.read_text(encoding="utf-8")) or {}


def _load_probe_ids() -> set[str]:
    ids: set[str] = set()
    for line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.add(json.loads(line)["id"])
    return ids


def _sdk_task_categories() -> dict[str, str]:
    """Parse TASKS dict from sdk_direct.py without importing it."""
    import ast
    src = SDK_DIRECT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TASKS":
                    tasks_dict = ast.literal_eval(node.value)
                    return {tid: v["category"] for tid, v in tasks_dict.items()}
    return {}


KNOWN_TASK_IDS = {
    "CH-01", "CH-02", "CN-01", "FO-01",
    "LH-01", "LH-02", "RE-01", "RE-02",
    "RH-01", "RH-02", "SH-01", "SH-02",
    "SO-01", "SW-01",
}


@pytest.fixture(scope="module")
def task_map():
    return _load_task_map()


@pytest.fixture(scope="module")
def probe_ids():
    return _load_probe_ids()


@pytest.fixture(scope="module")
def sdk_categories():
    return _sdk_task_categories()


def test_all_task_ids_present(task_map):
    missing = KNOWN_TASK_IDS - set(task_map.keys())
    assert not missing, f"task_map.yaml missing entries for: {sorted(missing)}"


def test_no_extra_task_ids(task_map):
    extra = set(task_map.keys()) - KNOWN_TASK_IDS
    assert not extra, f"task_map.yaml has unexpected task IDs: {sorted(extra)}"


def test_each_entry_has_probe_id_and_category(task_map):
    bad = [tid for tid, v in task_map.items() if not isinstance(v, dict) or not v.get("probe_id") or not v.get("category")]
    assert not bad, f"Entries missing probe_id or category: {bad}"


def test_all_probe_ids_exist_in_prompts(task_map, probe_ids):
    missing = [
        (tid, entry["probe_id"])
        for tid, entry in task_map.items()
        if entry.get("probe_id") not in probe_ids
    ]
    assert not missing, f"probe_ids not found in prompts.jsonl: {missing}"


def test_categories_match_sdk_direct(task_map, sdk_categories):
    mismatches = []
    for tid, entry in task_map.items():
        yaml_cat = entry.get("category")
        sdk_cat = sdk_categories.get(tid)
        if sdk_cat is not None and yaml_cat != sdk_cat:
            mismatches.append(f"{tid}: yaml={yaml_cat!r} sdk={sdk_cat!r}")
    assert not mismatches, "Category mismatches:\n" + "\n".join(mismatches)
