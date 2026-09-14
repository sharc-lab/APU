"""Focused tests for E-FILTER C2 amendments (docs/CURSOR_PROMPT_C2_efilter.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from seam.agent.document_corpus import (
    CHUNK_TOKEN_MAX,
    CHUNK_TOKEN_MIN,
    DocumentCorpus,
    proxy_tokens,
)
from seam.agent.harness import _estimate_prompt_tokens
from seam.agent.tools import ToolWorld, load_workload
from seam.analysis.efilter import deadline_grid, material_deadline_point
from seam.gitinfo import repo_root
from seam.hashing import sha256_file
from seam.tools.efilter_run import evaluate_pilot_context_gate

_ROOT = repo_root(Path(__file__).parent)


def test_corpus_determinism_and_token_band() -> None:
    a = DocumentCorpus(seed=20260804, n_chunks=48)
    b = DocumentCorpus(seed=20260804, n_chunks=48)
    assert [c.text for c in a.chunks] == [c.text for c in b.chunks]
    assert a.total_token_proxy == b.total_token_proxy
    assert a.total_token_proxy >= 20_000
    for chunk in a.chunks:
        assert CHUNK_TOKEN_MIN <= chunk.token_proxy <= CHUNK_TOKEN_MAX
        assert chunk.token_proxy == proxy_tokens(chunk.text)


def test_retrieve_documents_returns_1_to_3_chunks_in_size_range() -> None:
    corpus = DocumentCorpus(seed=20260804, n_chunks=48)
    world = ToolWorld(
        {
            "corpus_seed": 20260804,
            "corpus_n_chunks": 48,
            "directories": {},
            "files": {},
        }
    )
    assert world.corpus is not None
    seen_counts: set[int] = set()
    for i, query in enumerate(["latency", "KV bytes", "budget", "filter", "grid"]):
        text, err = world.execute("retrieve_documents", {"query": query})
        assert err is None
        payload = json.loads(text)
        assert 1 <= len(payload) <= 3
        seen_counts.add(len(payload))
        for item in payload:
            assert CHUNK_TOKEN_MIN <= item["token_proxy"] <= CHUNK_TOKEN_MAX
        # Same (seed, query, call_index) → identical.
        again = corpus.retrieve(query, call_index=i)
        assert [c.chunk_id for c in again] == [item["chunk_id"] for item in payload]
    assert seen_counts & {1, 2, 3}


def test_read_file_serves_large_corpus_docs_and_v1_world_stays_tiny() -> None:
    c2 = ToolWorld({"corpus_seed": 20260804, "corpus_n_chunks": 48, "files": {}})
    text, err = c2.execute("read_file", {"path": "/corpus/D003.txt"})
    assert err is None
    assert proxy_tokens(text) >= CHUNK_TOKEN_MIN

    v1 = ToolWorld(
        {"files": {"/docs/README.txt": "short"}, "directories": {"/docs": ["README.txt"]}}
    )
    tiny, err = v1.execute("read_file", {"path": "/docs/README.txt"})
    assert err is None
    assert tiny == "short"
    text, err = v1.execute("retrieve_documents", {"query": "x"})
    assert "unavailable" in text


def test_proxy_adds_scaffold_621() -> None:
    messages = [{"role": "user", "content": "abcd" * 10}]  # 40 chars → 10 tokens
    assert _estimate_prompt_tokens(messages) == 10
    assert _estimate_prompt_tokens(messages, scaffold_tokens=621) == 631


def test_deadline_grid_requires_at_least_8_points() -> None:
    t_preds = [10.0 + i * 0.5 for i in range(20)]
    grid = deadline_grid(t_preds, n=8, quantile_lo=0.0, quantile_hi=1.0, pad_factor=1.25)
    assert len(grid) == 8
    assert grid[0] < min(t_preds)
    assert grid[-1] > max(t_preds)
    with pytest.raises(Exception, match=">=8"):
        deadline_grid(t_preds, n=5, quantile_lo=0.0, quantile_hi=1.0, pad_factor=1.25)


def test_efilter_yaml_c2_pins_and_withdraws_8s() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == "efilter-c2"
    assert cfg["workload"]["max_steps"] == 16
    assert cfg["workload"]["n_tasks_pilot"] == 5  # C2b
    assert cfg["workload"]["max_tokens"] == 128  # C2b
    assert cfg["replay"]["n_deadlines"] >= 8
    assert cfg["replay"]["p95_step_target_s"] is None
    assert "withdrawn" in cfg["replay"]["p95_step_target_status"]
    assert cfg["policy"]["prompt_token_source"] == "chars_div_4_plus_scaffold"
    assert cfg["policy"]["prompt_token_scaffold_tokens"] == 621
    task_path = _ROOT / cfg["workload"]["task_list"]
    assert sha256_file(task_path) == cfg["workload"]["task_list_sha256"]
    workload = load_workload(task_path)
    assert len(workload.tasks) >= 20
    assert workload.world.corpus is not None


def test_am032_present_in_amendments() -> None:
    text = (_ROOT / "AMENDMENTS.md").read_text(encoding="utf-8")
    assert "## AM-032" in text
    assert "8 s" in text or "8s" in text
    assert "PRE-DATA" in text


def test_measurement_paging_exclusionary_for_efilter() -> None:
    measurement = yaml.safe_load(
        (_ROOT / "configs" / "measurement.yaml").read_text(encoding="utf-8")
    )
    assert measurement["paging"]["exclude_on_failure"] is True


def test_pilot_gate_logic_with_fake_growth_curves() -> None:
    # C2b: 20k peak gate withdrawn; median C_max/C_min >= 3.0.
    fail = evaluate_pilot_context_gate(
        {"max_context_tokens_observed": 50_000, "per_task_ratios": [1.2, 1.3, 1.4]},
        min_median_context_ratio=3.0,
    )
    assert fail["cleared"] is False
    assert fail["stop_message"] and "STOP" in fail["stop_message"]

    ok = evaluate_pilot_context_gate(
        {"max_context_tokens_observed": 5000, "per_task_ratios": [9.0, 7.9, 14.0]},
        min_median_context_ratio=3.0,
    )
    assert ok["cleared"] is True
    assert ok["stop_message"] is None
    assert ok["max_context_tokens_is_gate"] is False


def test_material_deadline_point_picks_loosest_material_ratio() -> None:
    curve = [
        {
            "deadline_s": 5.0,
            "n_surviving_steps": 2,
            "p95_actual_wall_s_surviving": 4.0,
            "over_provisioning": {"peak_kv_bytes_resident": {"point": 1.5}},
        },
        {
            "deadline_s": 8.0,
            "n_surviving_steps": 3,
            "p95_actual_wall_s_surviving": 6.0,
            "over_provisioning": {"peak_kv_bytes_resident": {"point": 1.3}},
        },
        {
            "deadline_s": 12.0,
            "n_surviving_steps": 4,
            "p95_actual_wall_s_surviving": 9.0,
            "over_provisioning": {"peak_kv_bytes_resident": {"point": 1.0}},
        },
    ]
    point = material_deadline_point(curve, materiality_ratio=1.2)
    assert point is not None
    assert point["deadline_s"] == 8.0
